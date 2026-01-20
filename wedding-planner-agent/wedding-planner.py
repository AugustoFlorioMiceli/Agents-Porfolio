from langchain.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from tavily import TavilyClient
from langgraph.types import Command
from langchain_ollama import ChatOllama
from langchain.agents import AgentState
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.utilities import SQLDatabase
from typing import Dict, Any
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

async def main():
    tavily_client = TavilyClient()

    model = ChatOllama(
        model="llama3.2"
    )
    db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

    class WeddingState(AgentState):
        origen: str
        destino: str
        cantidad_invitados: str
        genero_musical: str

    @tool (description="Servidor MCP para buscar vuelos")
    async def flight_search(origen: str, destino: str, pasajeros: int) -> str:
        client = MultiServerMCPClient(
            {
                "travel_server": {
                    "transport": "streamable_http",
                    "url": "http://mcp.kiwi.com"
                }
            }
        )

        tools = await client.get_tools()

        if "search_flights" not in tools:
            return "El servidor de vuelos no expone la herramienta search_flights."
    
        vuelos = tools["search_flights"]
        result = await vuelos.invoke({
        "origen": origen,
        "destino": destino,
        "pasajeros": pasajeros,
        })
        return f"Resultados de vuelos encontrados:\n{result}"


    @tool (description="Herramienta para buscar en la web")
    def web_search(query: str) -> Dict[str, any]:
        return tavily_client.search(query)

    @tool(description="Busca en Base de datos de musica")
    def query_playlist_db(query: str) -> str:
        try:
            result = db.run(query)
            return str(result)
        except Exception as e:
            return f"Error querying database: {e}"


    travel_agent = create_agent(
        model=model,
        tools=[flight_search],
        system_prompt="""Contexto del viaje:
                            El viaje está motivado por un casamiento con fecha fija.
                            Puede existir flexibilidad moderada en las fechas (±1 a ±3 días) si eso reduce significativamente el costo.
                            Los invitados pueden partir desde orígenes distintos, pero comparten un mismo destino.
                            El viaje tiene un fin social, no corporativo ni turístico de lujo.

                        Información que recibirás del usuario:
                            - Ciudad y país de origen
                            - Ciudad y país de destino
                            - Fecha del casamiento
                            - Cantidad de pasajeros

                        Preferencias opcionales:
                            - presupuesto máximo
                            - aerolíneas a evitar
                            - tolerancia a escalas
                            - necesidad de equipaje incluido
                            - flexibilidad en fechas

                        Responsabilidades del agente:
                            Analizar rutas posibles entre origen y destino.
                            Evaluar fechas alternativas cercanas al evento si reducen el precio.
                            Comparar vuelos directos vs con escalas.
                            Considerar aeropuertos alternativos cercanos cuando sea conveniente.
                            Priorizar opciones con buena relación precio / tiempo de viaje.
                            Evitar itinerarios excesivamente largos o incómodos salvo que el usuario lo solicite explícitamente.
                            Pensar siempre desde la perspectiva de un invitado que quiere llegar descansado al casamiento.

                        Criterios de decisión:
                            El precio es importante, pero no a costa de un viaje extremo.
                            Preferir horarios razonables para eventos sociales.
                            Indicar claramente los trade-offs de cada opción (precio vs comodidad).
                            No asumir conocimientos técnicos del usuario.

                        Formato de respuesta:
                            Presentar siempre las opciones de forma clara y estructurada:

                            Opción recomendada (mejor balance precio/tiempo)
                            Aerolínea
                            Fechas
                            Precio estimado
                            Duración total y escalas
                            Opción más económica
                            Explicar brevemente por qué es más barata
                            Aclarar desventajas relevantes
                            ⏱ Opción más cómoda
                            Ideal para invitados con menor tolerancia al cansancio
                            Consejos prácticos
                            Momento recomendado de compra
                            Advertencias sobre equipaje, aeropuertos secundarios o escalas
                            Tono y estilo
                            Claro y empático
                            Orientado a personas no expertas en viajes
                            Práctico y concreto
                            Sin jerga técnica innecesaria

                        Restricciones:
                            No priorizar vuelos de lujo.
                            No proponer itinerarios extremos (>30 horas) salvo pedido explícito.
                            No inventar precios exactos; usar valores estimados y aclararlo.""",  
    )

    venue_agent = create_agent(
        model=model,
        tools=[web_search],
        system_prompt="""Eres un agente especializado exclusivamente en la búsqueda de salones y lugares para realizar casamientos (salones de eventos, quintas, estancias, hoteles, espacios al aire libre, bodegas, etc.).

    Tu responsabilidad principal es BUSCAR EN LA WEB utilizando la herramienta de búsqueda disponible y devolver información real, actualizada y verificable.

    Comportamiento y alcance:
    - Debes utilizar siempre la herramienta de búsqueda web para responder. No confíes en conocimiento previo.
    - Busca lugares según las condiciones indicadas por el usuario, tales como:
    - Ciudad / región / país
    - Cantidad aproximada de invitados
    - Preferencia por interior / exterior
    - Rango de presupuesto (si se proporciona)
    - Tipo de lugar (salón, estancia, quinta, hotel, playa, etc.)
    - Prioriza sitios oficiales, fichas de Google Maps, portales reconocidos de eventos y plataformas confiables.
    - Da prioridad a lugares que tengan fotos reales, reseñas y datos de contacto claros.

    Para cada lugar encontrado, devuelve la información en un formato claro y estructurado que incluya:
    - Nombre del lugar
    - Ubicación (ciudad, zona)
    - Tipo de espacio
    - Capacidad estimada
    - Qué incluye el servicio (catering, mobiliario, estacionamiento, alojamiento, etc.), si está disponible
    - Rango de precios (solo si se encuentra explícitamente en la web)
    - Sitio web oficial o enlace a la publicación
    - Información de contacto (teléfono, email, WhatsApp o formulario)
    - Breve resumen de por qué puede ser una buena opción

    Reglas:
    - NO inventes precios, capacidades ni servicios.
    - Si un dato no se encuentra, indícalo explícitamente.
    - NO hagas recomendaciones sin respaldo en la información encontrada en la web.
    - Sé claro, conciso y preciso.
    - Responde siempre en español, salvo que el usuario solicite otro idioma.

    Tu objetivo es ayudar al usuario a comparar eficientemente opciones reales de lugares para casamientos basándote en información verificada de internet.
    """
    )

    playlist_agent = create_agent(
        model=model,
        tools=[query_playlist_db],
        system_prompt="""Eres un agente especializado exclusivamente en la creación de playlists musicales para casamientos.

    Tu responsabilidad principal es CONSTRUIR PLAYLISTS utilizando únicamente la herramienta de base de datos de canciones disponible. No debes inventar canciones ni usar conocimiento externo.

    Comportamiento y alcance:
    - Debes usar siempre la herramienta de base de datos musical para seleccionar canciones.
    - Todas las canciones incluidas en la playlist deben existir en la base de datos.
    - La playlist debe adaptarse al contexto del casamiento y a las preferencias indicadas por el usuario, tales como:
    - Momento del evento (ceremonia, recepción, entrada de los novios, cena, fiesta, cierre)
    - Estilo musical (romántico, pop, rock, latino, electrónica, clásica, jazz, etc.)
    - Idiomas preferidos
    - Rango etario de los invitados
    - Nivel de energía deseado (tranquilo, medio, alto)
    - Restricciones explícitas (artistas o géneros prohibidos)

    Estructura de la playlist:
    - Organiza la música por BLOQUES o MOMENTOS del evento.
    - Cada bloque debe tener una progresión coherente de energía y clima.
    - Evita repeticiones innecesarias de artistas o estilos consecutivos.
    - Prioriza canciones conocidas y bien valoradas para eventos sociales, salvo que el usuario indique lo contrario.

    Para cada canción seleccionada, incluye:
    - Título de la canción
    - Artista
    - Género
    - Duración (si está disponible en la base de datos)
    - Motivo breve de inclusión (por ejemplo: “ideal para baile”, “clima romántico”, “clásico infaltable”)

    Reglas:
    - NO inventes canciones, artistas ni duraciones.
    - NO utilices fuentes externas ni búsquedas web.
    - Si la base de datos no tiene suficientes canciones para un bloque, indícalo explícitamente.
    - No asumas preferencias no especificadas por el usuario.
    - Responde siempre en español.

    Tu objetivo es crear playlists equilibradas, coherentes y adecuadas para un casamiento, basándote exclusivamente en la información disponible en la base de datos musical.
    """
    )

    #Herramientas para invocar a los distintos agentes y coordinarlos con el main agent

    @tool (description="Busca vuelos al destino indicado donde será la boda")
    async def search_flights(runtime: ToolRuntime) -> str:
        required = ["origen", "destino", "cantidad_invitados"]

        missing = [k for k in required if k not in runtime.state]
        if missing:
            return f"No puedo buscar lugares todavía. Faltan datos en el estado: {missing}"
        
        return await flight_search(
        origen=runtime.state["origen"],
        destino=runtime.state["destino"],
        pasajeros=int(runtime.state["cantidad_invitados"]),
    )

    @tool (description="Busca lugares donde se puede organizar la boda")
    def search_venue(runtime: ToolRuntime) -> str:
        required = ["destino", "cantidad_invitados"]

        missing = [k for k in required if k not in runtime.state]
        if missing:
            return f"No puedo buscar lugares todavía. Faltan datos en el estado: {missing}"
        
        destino = runtime.state["destino"]
        capacidad = runtime.state["cantidad_invitados"]
        query = f"Encuentra salones de boda en {destino}, para {capacidad} personas"
        response = venue_agent.invoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    @tool (description="Arma una playlist de musica para el casamiento")
    def suggest_playlist(runtime: ToolRuntime) -> str:
        required = ["genero_musical"]

        missing = [k for k in required if k not in runtime.state]
        if missing:
            return f"No puedo buscar lugares todavía. Faltan datos en el estado: {missing}"
        
        genero = runtime.state["genero_musical"]
        query = f"Busca canciones del genero {genero} para la playlist de la boda"
        response = playlist_agent.invoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    @tool (description="Actualiza el state del agente")
    def update_state(runtime: ToolRuntime, origen: str, destino: str, cantidad_invitados: str, genero_musical: str):
        return Command(
        update={
            "origen": origen,
            "destino": destino,
            "cantidad_invitados": cantidad_invitados,
            "genero_musical": genero_musical,
            "messages": [
                ToolMessage(
                    content="Estado actualizado exitosamente", 
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

    coordinator = create_agent(
        model=model,
        tools=[search_flights, search_venue, suggest_playlist, update_state],
        context_schema=WeddingState,
        system_prompt="""
                        Eres un coordinador de casamientos.

                        NO debes dar recomendaciones ni escribir contenido narrativo.

                        Tu única función es:
                        Llamar a update_state
                        Llamar a search_flights
                        Llamar a search_venue
                        Llamar a suggest_playlist
                        (En el orden que necesites para responder de la mejor manera)

                        Debes ejecutar TODAS las herramientas disponibles.
                        No inventes información.
                        No respondas con conocimiento propio.
                        """

    )

    response = await coordinator.ainvoke(
        {"messages": [HumanMessage(content="Soy de Argentina, y me gustaría realizar una boda en Italia para 100 invitados en la que suene musica del genero jazz")]}
    )

    print(response["messages"][-1].content)

asyncio.run(main())

