from flask import Flask, render_template_string, request
import json
import numpy as np
from algo import get_graph_from_binary_matrix, backtracking_dfs, greedy_dfs, forced_move_dfs, edge_elimination_dfs, validation_forced_move_dfs, validation_edge_elimination_dfs
from image import ImageProcessor
import base64
import cv2 as cv
import os

public_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'public'))
app = Flask(__name__, static_folder=public_path, static_url_path='/public')
# app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

processor = ImageProcessor()


# Base HTML with placeholder tokens for which sections/buttons are active initially
BASE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Block Fill Solver</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        header h1 { font-size: 2.5em; margin-bottom: 10px; }
        header p { font-size: 1.1em; opacity: 0.9; }
        
        .content { padding: 40px; }
        
        .mode-selector {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            justify-content: center;
        }
        
        .mode-btn {
            flex: 1;
            max-width: 300px;
            padding: 20px;
            border: 3px solid #ddd;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
            text-align: center;
            background: white;
        }
        
        .mode-btn:hover {
            border-color: #667eea;
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        
        .mode-btn.active {
            border-color: #667eea;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .mode-btn h3 { margin-bottom: 10px; font-size: 1.5em; }
        .mode-btn p { font-size: 0.9em; opacity: 0.8; }
        
        .upload-section, .manual-section {
            display: none;
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        
        .upload-section.active, .manual-section.active { display: block; }
        
        .form-group { margin-bottom: 20px; margin-top: 20px; }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        input[type="file"] {
            display: block;
            width: 100%;
            padding: 12px;
            border: 2px dashed #667eea;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        input[type="file"]:hover {
            border-color: #764ba2;
            background: #f8f9fa;
        }
        
        input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        
        select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            background: white;
            font-size: 16px;
            cursor: pointer;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .grid-editor {
            margin-top: 20px;
            display: none;
        }
        
        .grid-editor.active { display: block; }
        
        .grid-container {
            display: inline-block;
            border: 2px solid #333;
            margin: 20px auto;
            background: #fff;
        }
        
        .grid-row { display: flex; }
        
        .grid-cell {
            width: 40px;
            height: 40px;
            border: 1px solid #ccc;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .grid-cell.walkable { background: #e0e0e0; }
        .grid-cell.obstacle { background: #2f251e; }
        .grid-cell.start { background: #71d63b; }
        .grid-cell.finish { background: #ff0000; }
        
        .grid-cell:hover { transform: scale(1.1); z-index: 10; }
        
        .grid-controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            justify-content: center;
        }
        
        .control-btn {
            padding: 10px 20px;
            border: 2px solid #667eea;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .control-btn.active {
            background: #667eea;
            color: white;
        }
        
        .info-box {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #2196F3;
            display: flex;
            gap: 30px;
            align-items: flex-start;
        }

        .info-text {
            flex: 1;
            min-width: 0;
        }
        
        .info-box h4 { color: #1976D2; margin-bottom: 10px; }
        .info-box p { color: #555; line-height: 1.6; }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .alert-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .alert-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .results-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 40px;
        }
        
        .result-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        
        .result-box h3 { margin-bottom: 15px; color: #333; }
        
        .result-box img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .example-card {
            background: #fff;
            border: 1px solid rgba(0,0,0,0.06);
            padding: 16px;
            border-radius: 8px;
            box-shadow: 0 6px 18px rgba(12,18,24,0.06);
            text-align: center;
            flex-shrink: 0;
            width: 280px;
        }

        .example-card .example-title {
            font-weight: 600;
            text-align: center;
            margin-bottom: 12px;
            font-size: 0.9em;
            color: #333;
        }

        .example-card img {
            display: block;
            width: 100%;
            max-width: 200px;
            height: auto;
            border-radius: 6px;
            margin: 0 auto;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        @media (max-width: 768px) {
            .results-section { grid-template-columns: 1fr; }
            .mode-selector { flex-direction: column; }
            header h1 { font-size: 2em; }
            .content { padding: 20px; }
            
            .info-box {
                flex-direction: column;
            }
            
            .example-card {
                width: 100%;
                max-width: 100%;
            }
        }

        @media (max-width: 720px) {
            .upload-example {
                grid-template-columns: 1fr;
            }

            .example-card { 
                padding: 12px; 
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 Block Fill Solver</h1>
            <p>Upload your puzzle image or create a custom matrix</p>
        </header>
        
        <div class="content">
            {% if error %}
            <div class="alert alert-error">
                <strong>Error:</strong> {{ error }}
            </div>
            {% endif %}
            
            {% if success %}
            <div class="alert alert-success">
                <strong>Success!</strong> {{ success }}
            </div>
            {% endif %}
            
            <div class="mode-selector">
                <div class="mode-btn __UPLOAD_BTN_ACTIVE__" onclick="switchMode('upload')">
                    <h3>📤 Upload Image</h3>
                    <p>Upload a puzzle screenshot</p>
                </div>
                <div class="mode-btn __MANUAL_BTN_ACTIVE__" onclick="switchMode('manual')">
                    <h3>✏️ Manual Matrix</h3>
                    <p>Create your own puzzle</p>
                </div>
            </div>
            
            <div class="upload-section __UPLOAD_ACTIVE__">
                <div class="info-box">
                    <div class="info-text">
                        <h4>📝 Upload Instructions:</h4>
                        <p>
                            Upload a screenshot of a grid puzzle. The system will automatically detect:<br>
                            • Grid cells and layout<br>
                            • Start positions<br>
                            • Walkable paths
                        </p>
                    </div>

                    <div class="example-card">
                        <div class="example-title">Valid example: </div>
                        <img src="{{ url_for('static', filename='test3.png') }}"
                            alt="Example input with only a start point. No finish point present."
                            title="Example: only start">
                    </div>
                </div>
                
                <form method="POST" action="/solve_upload" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="file">📁 Select Puzzle Image:</label>
                        <input type="file" id="file" name="file" accept=".png,.jpg,.jpeg" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="algorithm">🧮 Select Algorithm:</label>
                        <select id="algorithm" name="algorithm" required>
                            <option value="backtracking" selected>Backtracking DFS</option>
                            <option value="greedy" selected>Warnsdorff's rule</option>
                            <option value="validation_edge_elimination" selected>Validation Edge Elimination</option>
                            <option value="validation_forced_move" selected>Validation Forced Move</option>
                            <option value="edge_elimination" selected>Edge Elimination</option>
                            <option value="forced_move" selected>Forced Move</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn">🚀 Solve Puzzle</button>
                </form>
            </div>
            
            <!-- Manual Mode -->
            <div class="manual-section __MANUAL_ACTIVE__">
                <div class="info-box">
                <div class="info-text">
                    <h4>✏️ Manual Instructions:</h4>
                    <p>
                        1. Set grid dimensions<br>
                        2. Click cells to set: Start (Green), Obstacles (Dark), Walkable (Grey)<br>
                        3. Solve your custom puzzle!
                    </p>
                </div>
                </div>
                
                <form id="manualForm">
                    <div class="form-group">
                        <label>Grid Size:</label>
                        <div style="display: flex; gap: 10px;">
                            <input type="number" id="rows" placeholder="Rows" min="3" max="20" value="5" style="width: 48%;">
                            <input type="number" id="cols" placeholder="Cols" min="3" max="20" value="5" style="width: 48%;">
                        </div>
                    </div>
                    
                    <button type="button" class="btn" onclick="createGrid()">🎨 Create Grid</button>
                </form>
                
                <div class="grid-editor" id="gridEditor">
                    <div class="grid-controls">
                        <button class="control-btn" onclick="setMode('start')">🟢 Start</button>
                        <button class="control-btn" onclick="setMode('obstacle')">⬛ Obstacle</button>
                        <button class="control-btn active" onclick="setMode('walkable')">⬜ Walkable</button>
                    </div>
                    
                    <div style="text-align: center;">
                        <div id="gridContainer" class="grid-container"></div>
                    </div>
                    
                    <form method="POST" action="/solve_manual">
                        <input type="hidden" id="matrixData" name="matrix_data">
                        <div class="form-group">
                            <label for="algorithm2">🧮 Select Algorithm:</label>
                            <select id="algorithm2" name="algorithm">
                                <option value="backtracking" selected>Backtracking DFS</option>
                                <option value="greedy" selected>Warnsdorff's rule</option>
                                <option value="validation_edge_elimination" selected>Validation Edge Elimination</option>
                                <option value="validation_forced_move" selected>Validation Forced Move</option>
                                <option value="edge_elimination" selected>Edge Elimination</option>
                                <option value="forced_move" selected>Forced Move</option>
                            </select>
                        </div>
                        <button type="submit" class="btn" onclick="return submitMatrix()">🚀 Solve Puzzle</button>
                    </form>
                </div>
            </div>
            
            {% if original_img and result_img %}
            {% if path_length %}
            <div class="info-box" style="margin-top: 20px;">
                <div class="info-text">
                <h4>📊 Solution Statistics:</h4>
                <p>
                    <strong>Path Length:</strong> {{ path_length }} steps<br>
                    <strong>Algorithm:</strong> {{ algo_used }}<br>
                    <strong>Time Elapsed:</strong> {{time_elapsed}}<br>
                    <strong>Status:</strong> Solution found successfully!
                </p>
            </div>
            </div>

            <div class="results-section">
                <div class="result-box">
                    <h3>📷 Original Image</h3>
                    <img src="data:image/png;base64,{{ original_img }}" alt="Original">
                </div>
                
                <div class="result-box">
                    <h3>✅ Solution Path</h3>
                    <img src="data:image/png;base64,{{ result_img }}" alt="Solution">
                </div>
            </div>
            
            {% endif %}
            {% endif %}
        </div>
    </div>
    
    <script>
        let currentMode = 'walkable';
        let gridMatrix = [];
        
        function switchMode(mode) {
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            // set button active properly without relying on nth-child
            if (mode === 'upload') {
                document.querySelectorAll('.mode-btn')[0].classList.add('active');
            } else {
                document.querySelectorAll('.mode-btn')[1].classList.add('active');
            }
            
            document.querySelector('.upload-section').classList.toggle('active', mode === 'upload');
            document.querySelector('.manual-section').classList.toggle('active', mode === 'manual');
        }
        
        function createGrid() {
            const rows = parseInt(document.getElementById('rows').value);
            const cols = parseInt(document.getElementById('cols').value);
        
            if (rows < 1 || cols < 1) {
                alert('Grid size must be at least 1');
                return;
            }
            
            // Initialize matrix with all 1s (walkable)
            gridMatrix = Array(rows).fill().map(() => Array(cols).fill(1));
            
            const container = document.getElementById('gridContainer');
            container.innerHTML = '';
            
            for (let i = 0; i < rows; i++) {
                const row = document.createElement('div');
                row.className = 'grid-row';
                
                for (let j = 0; j < cols; j++) {
                    const cell = document.createElement('div');
                    cell.className = 'grid-cell walkable';
                    cell.dataset.row = i;
                    cell.dataset.col = j;
                    cell.onclick = () => toggleCell(i, j);
                    row.appendChild(cell);
                }
                
                container.appendChild(row);
            }
            
            document.getElementById('gridEditor').classList.add('active');
        }
        
        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.control-btn').forEach(btn => btn.classList.remove('active'));
            // safe approach: find the clicked element via event
            if (event && event.target) {
                // if the button contains an inner element (emoji/text), climb up to button
                let el = event.target;
                while (el && !el.classList.contains('control-btn')) {
                    el = el.parentElement;
                }
                if (el) el.classList.add('active');
            }
        }
        
        function toggleCell(row, col) {
            const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
            if (!cell) return;
            
            // Remove existing start/finish if setting new one
            if (currentMode === 'start') {
                document.querySelectorAll('.grid-cell.start').forEach(c => {
                    c.className = 'grid-cell walkable';
                    const r = parseInt(c.dataset.row);
                    const cl = parseInt(c.dataset.col);
                    gridMatrix[r][cl] = 1;
                });
                cell.className = 'grid-cell start';
                gridMatrix[row][col] = 2;
            } else if (currentMode === 'obstacle') {
                cell.className = 'grid-cell obstacle';
                gridMatrix[row][col] = 0;
            } else {
                cell.className = 'grid-cell walkable';
                gridMatrix[row][col] = 1;
            }
        }
        
        function submitMatrix() {
            const hasStart = gridMatrix.some(row => row.includes(2));

            if (!hasStart) {
                alert('Please set both Start (Green) and Finish (Red) positions!');
                return false;
            }
            
            document.getElementById('matrixData').value = JSON.stringify(gridMatrix);
            return true;
        }
    </script>
</body>
</html>
'''

# Build two concrete templates by replacing tokens:
IMAGE_TEMPLATE = BASE_HTML.replace('__UPLOAD_ACTIVE__', 'active') \
                          .replace('__MANUAL_ACTIVE__', '') \
                          .replace('__UPLOAD_BTN_ACTIVE__', 'active') \
                          .replace('__MANUAL_BTN_ACTIVE__', '')

MANUAL_TEMPLATE = BASE_HTML.replace('__UPLOAD_ACTIVE__', '') \
                           .replace('__MANUAL_ACTIVE__', 'active') \
                           .replace('__UPLOAD_BTN_ACTIVE__', '') \
                           .replace('__MANUAL_BTN_ACTIVE__', 'active')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def img_to_datauri_b64(img_ndarray):
    """Encode BGR numpy image to PNG base64 string (no files on disk)."""
    _, buff = cv.imencode('.png', img_ndarray)
    return base64.b64encode(buff).decode('utf-8')

@app.route('/', methods=['GET'])
def index():
    # default to upload view
    return render_template_string(IMAGE_TEMPLATE)


@app.route('/solve_upload', methods=['POST'])
def solve_upload():
    if 'file' not in request.files:
        return render_template_string(IMAGE_TEMPLATE, error='No file uploaded')
    
    file = request.files['file']
    algorithm = request.form.get('algorithm', 'edge_elimination')
    
    if file.filename == '':
        return render_template_string(IMAGE_TEMPLATE, error='No file selected')
    
    if file and allowed_file(file.filename):
        try:
            raw = file.read()
            img = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)

            if img is None:
                return render_template_string(IMAGE_TEMPLATE, error='Could not read uploaded image')
        
            matrix = processor.img_to_matrix(img)

            processor.generate_img(matrix)

            # Convert to graph
            G, start = get_graph_from_binary_matrix(matrix)
            
            if start is None :
                return render_template_string(IMAGE_TEMPLATE, 
                    error='Could not find start or finish cell. Make sure the image has clear grid structure.')
            
            # Run algorithm
            if algorithm == 'backtracking' :
                path, finish_status, finish_node, time_elapsed = backtracking_dfs(G, start)
                algo_name = 'Backtracking DFS'
            elif algorithm == 'greedy'  :
                path, finish_status, finish_node, time_elapsed = greedy_dfs(G, start)
                algo_name = 'Greedy DFS'
            elif algorithm == 'forced_move' :
                path, finish_status, finish_node, time_elapsed = forced_move_dfs(G, start)
                algo_name = 'Forced Move DFS'
            elif algorithm == 'edge_elimination' :
                path, finish_status, finish_node, time_elapsed = edge_elimination_dfs(G, start)
                algo_name = 'Edge Elimination DFS'
            elif algorithm == 'validation_forced_move' :
                path, finish_status, finish_node, time_elapsed = validation_forced_move_dfs(G, start)
                algo_name = 'Validation Forced Move DFS'
            elif algorithm == 'validation_edge_elimination' :
                path, finish_status, finish_node, time_elapsed = validation_edge_elimination_dfs(G, start)
                algo_name = 'Validation Edge Elimination DFS'
            else:
                return render_template_string(IMAGE_TEMPLATE, error='Unknown algorithm')
            
            if finish_status is False:
                return render_template_string(IMAGE_TEMPLATE, 
                    error='Could not find path from start to finish.')
            
            result_img = processor.draw_path_on_image(matrix, path, start, finish_node)
            original_b64 = img_to_datauri_b64(getattr(processor, 'original_img_bgr', processor.last_img_bgr))
            result_b64 = img_to_datauri_b64(result_img)
            
            return render_template_string(IMAGE_TEMPLATE, 
                original_img=original_b64,
                result_img=result_b64,
                path_length=len(path),
                algo_used=algo_name,
                time_elapsed=time_elapsed,
                success='Puzzle solved successfully!')
            
        except Exception as e:
            return render_template_string(IMAGE_TEMPLATE, error=f'Error: {str(e)}')
    
    return render_template_string(IMAGE_TEMPLATE, error='Invalid file type')


@app.route('/solve_manual', methods=['POST'])
def solve_manual():
    try:
        matrix_json = request.form.get('matrix_data')
        algorithm = request.form.get('algorithm', 'edge_elimination')
        
        if not matrix_json:
            return render_template_string(MANUAL_TEMPLATE, error='No matrix data received')
        
        matrix = np.array(json.loads(matrix_json))

        generated = processor.generate_img(matrix)

        processor.img_to_matrix(generated)

        # Convert to graph
        G, start = get_graph_from_binary_matrix(matrix)
        
        if start is None:
            return render_template_string(MANUAL_TEMPLATE, 
                error='Could not find start or finish in matrix')
        
        if algorithm == 'backtracking' :
            path, finish_status, finish_node, time_elapsed = backtracking_dfs(G, start)
            algo_name = 'Backtracking DFS'
        elif algorithm == 'greedy'  :
            path, finish_status, finish_node, time_elapsed = greedy_dfs(G, start)
            algo_name = 'Greedy DFS'
        elif algorithm == 'forced_move' :
            path, finish_status, finish_node, time_elapsed = forced_move_dfs(G, start)
            algo_name = 'Forced Move DFS'
        elif algorithm == 'edge_elimination' :
            path, finish_status, finish_node, time_elapsed = edge_elimination_dfs(G, start)
            algo_name = 'Edge Elimination DFS'
        elif algorithm == 'validation_forced_move' :
            path, finish_status, finish_node, time_elapsed = validation_forced_move_dfs(G, start)
            algo_name = 'Validation Forced Move DFS'
        elif algorithm == 'validation_edge_elimination' :
            path, finish_status, finish_node, time_elapsed = validation_edge_elimination_dfs(G, start)
            algo_name = 'Validation Edge Elimination DFS'
        else:
            return render_template_string(IMAGE_TEMPLATE, error='Unknown algorithm')
        
        if finish_status is False:
            return render_template_string(MANUAL_TEMPLATE, 
                error='Could not find path from start to finish.')
        
        result_img = processor.draw_path_on_image(matrix, path, start, finish_node)
        original_b64 = img_to_datauri_b64(getattr(processor, 'original_img_bgr', processor.last_img_bgr))
        result_b64 = img_to_datauri_b64(result_img)
        
        return render_template_string(MANUAL_TEMPLATE, 
            original_img=original_b64,
            result_img=result_b64,
            path_length=len(path),
            algo_used=algo_name,
            time_elapsed=time_elapsed,
            success='Custom puzzle solved successfully!')
        
    except Exception as e:
        return render_template_string(MANUAL_TEMPLATE, error=f'Error: {str(e)}')

# Local testing

# if __name__ == '__main__':
#     print("🚀 Starting Block Fill Solver...")
#     print("📍 Open your browser and go to: http://localhost:5000")
#     app.run(debug=True, host='0.0.0.0', port=5000)