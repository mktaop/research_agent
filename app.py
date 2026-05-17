import streamlit as st
import json, os
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List


st.set_page_config(page_title="AI Market Research Agent", layout="wide")
st.title("📊 AI Market Research Agent")
st.subheader("Powered by Gemini & Google Search")

api_key = os.environ.get("GOOGLE_API_KEY")

subject = st.text_input("What subject do you want market analysis for? And what calendar years?", 
                       placeholder="e.g., Plant-based milk in North America 2024-2026")

if st.button("Generate Analysis") and subject:
    if not api_key:
        st.error("Please provide API Key.")
    else:
        client = genai.Client(api_key=api_key)
        
        with st.status("Analyzing Market...", expanded=True) as status:
            st.write("Searching the web for current trends...")
            search_tool = types.Tool(
                                        google_search=types.GoogleSearch()
                                       )
            
            prompt_search = f"""
            Search the web for market trends for {subject}. 
            I need to know the market size, key players and their market share, 
            and primary consumer drivers.
            """
            
            response = client.models.generate_content(
                model="gemini-3-flash-preview", 
                contents=prompt_search,
                config=types.GenerateContentConfig(tools=[search_tool],
                                                   thinking_config=types.ThinkingConfig(thinking_level="high",
                                                                                       include_thoughts=True))
            )
            
            market_trends = response.text
            citations = []
            if response.candidates[0].grounding_metadata.grounding_chunks:
                for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                    if chunk.web:
                        citations.append({
                            "title": chunk.web.title,
                            "uri": chunk.web.uri
                        })

            st.write("Extracting data for visualizations...")
            class ChartConfig(BaseModel):
                type: str  # "bar" or "line"
                labels: List[str]
                data: List[float]
                label: str
                colors: List[str]
            
            class ChartList(BaseModel):
                chart_configurations: List[ChartConfig]
            
            prompt_charts = f"""
            Given the following market trends text, come up with a list of 1-3 meaningful bar or line charts 
            and generate chart data in JSON format.
            Market Trends: {market_trends}
            """
            
            chart_response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt_charts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_level="high"),
                    response_schema=ChartList  # passing the Pydantic class here
                )
            )
            
            chart_data = json.loads(chart_response.text)
            status.update(label="Analysis Complete!", state="complete")

        st.divider()
        st.write("Compiling final report with citations...")
    
        prompt_report = f"""
        You are an expert financial and market analyst. Generate a professional market analysis report in HTML.
        
        DATA:
        Trends: {market_trends}
        Charts: {chart_data}
        Sources: {json.dumps(citations)}
        
        STYLE INSTRUCTIONS:
        1. Write the main body using the Market Trends.
        2. Use a clean sans-serif font (Arial/Helvetica).
        2. Chart Sizing: Wrap every <canvas> in a div with `style="position: relative; height:300px; width:100%; max-width:700px; margin: auto;"`.
        3. Chart.js Config: In the Javascript 'options' for every chart, set `maintainAspectRatio: false` and `responsive: true`.
        4. Colors: Use a professional palette (e.g., #2c3e50, #2980b9).
        5. IMPORTANT: Create a "Sources & References" section at the end of the HTML.
        6. Format the sources as an ordered list with clickable hyperlinks using the provided Title and URI.
        7. Return ONLY the HTML code.
        """
        
        report_response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt_report,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="high")),
        )
        
        html_content = report_response.text.replace("```html", "").replace("```", "")
        st.components.v1.html(html_content, height=800, scrolling=True)

        st.download_button("Download Report (HTML)", html_content, file_name="market_report.html")
