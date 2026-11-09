import sys
import re
from pathlib import Path

# 1. Update routes.py
routes_path = Path("app/api/routes.py")
content = routes_path.read_text(encoding="utf-8")

if "def analyze_csv" not in content:
    imports = """from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import pandas as pd
import io
import base64
from app.services.visualization_service import analyze_and_plot
from app.services.tts_service import generate_speech
"""
    # Replace existing FastAPI imports to avoid duplication if possible, or just insert
    content = content.replace("from fastapi import APIRouter, File, UploadFile, HTTPException\n", imports)
    
    endpoint = """
@router.post("/analyze_csv")
async def analyze_csv(
    file: UploadFile = File(...),
    time_col: str = Form(...),
    val_col: str = Form(...)
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), comment="#")
        
        # Call the visualization service
        fig, explanation = analyze_and_plot(df, time_col, val_col)
        
        # Calculate stats for speech
        valid_data = df.dropna(subset=[val_col])
        if not valid_data.empty:
            max_val = valid_data[val_col].max()
            min_val = valid_data[val_col].min()
            avg_val = valid_data[val_col].mean()
            spoken_text = (
                f"Here is the chart for {val_col} over {time_col}. "
                f"The maximum value is {max_val:.2f}, "
                f"the minimum value is {min_val:.2f}, "
                f"and the average value is {avg_val:.2f}."
            )
        else:
            spoken_text = "I couldn't find any valid numerical data to analyze."
            
        # Generate TTS audio
        audio_bytes, content_type = await asyncio.to_thread(generate_speech, spoken_text)
        
        return {
            "fig_json": fig.to_json(),
            "explanation": explanation,
            "audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
            "content_type": content_type
        }
    except Exception as e:
        logger.error(f"Analyze CSV error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
"""
    content += endpoint
    routes_path.write_text(content, encoding="utf-8")
    print("Updated routes.py")


# 2. Update voice_ui.html
html_path = Path("frontend/voice_ui.html")
html = html_path.read_text(encoding="utf-8")

if "plotly" not in html:
    html = html.replace("</head>", '    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n</head>')

# Add chart generation UI
chart_ui = """
            <div id="card-csv-details" class="csv-summary" style="display:none;">
                <span class="csv-badge">CSV DATA</span>
                <span id="csv-shape-info"></span>
                <div class="column-tags" id="csv-columns-list"></div>
                
                <div id="chart-controls" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #444;">
                    <div style="margin-bottom: 5px; color: #8ed0ff; font-weight: bold;">Generate Insight Chart:</div>
                    <select id="x-axis-select" style="background: #252a36; color: white; border: 1px solid #444; border-radius: 4px; padding: 4px; margin-right: 5px; width: 120px;">
                        <option value="">X-Axis...</option>
                    </select>
                    <select id="y-axis-select" style="background: #252a36; color: white; border: 1px solid #444; border-radius: 4px; padding: 4px; margin-right: 5px; width: 120px;">
                        <option value="">Y-Axis...</option>
                    </select>
                    <button onclick="generateChart()" class="upload-btn" style="margin-top: 5px; background: #0055ff; border-color: #0044cc;">📊 Generate</button>
                </div>
            </div>
"""
# Replace existing card-csv-details
html = re.sub(
    r'<div id="card-csv-details" class="csv-summary" style="display:none;">.*?</div>\s*</div>',
    chart_ui + "        </div>",
    html,
    flags=re.DOTALL
)

# Add JS logic for parsing and generating chart
js_logic = """
                        csvColsList.innerHTML = "";
                        const xSelect = document.getElementById('x-axis-select');
                        const ySelect = document.getElementById('y-axis-select');
                        xSelect.innerHTML = '<option value="">X-Axis...</option>';
                        ySelect.innerHTML = '<option value="">Y-Axis...</option>';

                        if (data.column_names && Array.isArray(data.column_names)) {
                            data.column_names.forEach(col => {
                                const tag = document.createElement("span");
                                tag.className = "column-tag";
                                tag.innerText = col;
                                csvColsList.appendChild(tag);
                                
                                const optX = document.createElement("option");
                                optX.value = col;
                                optX.innerText = col;
                                xSelect.appendChild(optX);
                                
                                const optY = document.createElement("option");
                                optY.value = col;
                                optY.innerText = col;
                                ySelect.appendChild(optY);
                            });
                        }
"""
# Inject into uploadDocument
html = html.replace('csvColsList.innerHTML = "";\n                        if (data.column_names && Array.isArray(data.column_names)) {', js_logic + '\n                        if (false) {')

# Add the generateChart function
generate_func = """
        async function generateChart() {
            const fileInput = document.getElementById('file-upload');
            const file = fileInput.files[0];
            const xCol = document.getElementById('x-axis-select').value;
            const yCol = document.getElementById('y-axis-select').value;
            
            if (!file || !xCol || !yCol) {
                alert("Please select a file, X-axis, and Y-axis.");
                return;
            }

            statusEl.innerText = "Analyzing data and generating chart...";
            
            const formData = new FormData();
            formData.append("file", file);
            formData.append("time_col", xCol);
            formData.append("val_col", yCol);

            try {
                const res = await fetch(`${BACKEND_URL}/analyze_csv`, { method: "POST", body: formData });
                const data = await res.json();

                if (res.ok) {
                    statusEl.innerText = "Chart generated successfully!";
                    
                    // Add explanation to chat
                    const aiMsg = document.createElement('div');
                    aiMsg.className = 'ai-msg';
                    aiMsg.innerHTML = '<strong>System:</strong><br/>' + data.explanation;
                    chatLog.appendChild(aiMsg);
                    
                    // Create Plotly div
                    const plotDiv = document.createElement('div');
                    plotDiv.style.width = "100%";
                    plotDiv.style.height = "350px";
                    plotDiv.style.marginTop = "10px";
                    plotDiv.style.marginBottom = "20px";
                    plotDiv.id = 'plot-' + Date.now();
                    chatLog.appendChild(plotDiv);
                    
                    chatLog.scrollTop = chatLog.scrollHeight;
                    
                    // Render Plotly chart
                    const fig = JSON.parse(data.fig_json);
                    Plotly.newPlot(plotDiv.id, fig.data, fig.layout, {responsive: true});
                    
                    chatLog.scrollTop = chatLog.scrollHeight;

                    // Play audio
                    if (data.audio_base64) {
                        const audioBlob = b64toBlob(data.audio_base64, data.content_type);
                        const audioUrl = URL.createObjectURL(audioBlob);
                        if (currentAudio) currentAudio.pause();
                        currentAudio = new Audio(audioUrl);
                        currentAudio.play();
                    }
                } else {
                    statusEl.innerText = "Error: " + (data.detail || "Analysis failed");
                }
            } catch (err) {
                console.error(err);
                statusEl.innerText = "Failed to generate chart.";
            }
        }
        
        function b64toBlob(b64Data, contentType='', sliceSize=512) {
            const byteCharacters = atob(b64Data);
            const byteArrays = [];
            for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
                const slice = byteCharacters.slice(offset, offset + sliceSize);
                const byteNumbers = new Array(slice.length);
                for (let i = 0; i < slice.length; i++) {
                    byteNumbers[i] = slice.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                byteArrays.push(byteArray);
            }
            return new Blob(byteArrays, {type: contentType});
        }
"""
html = html.replace('</script>', generate_func + '\n    </script>')
html_path.write_text(html, encoding="utf-8")
print("Updated voice_ui.html")
