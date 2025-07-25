#!/usr/bin/env python3
"""
Generate HTML visualization reports for each position of interest from step 5.

This script creates an HTML file for each analyzed position, displaying:
- Interactive Go board automatically positioned at the move of interest
- Complete NMF analysis data
- Go pattern analysis
- Component comparison data

Requirements: The Go board must automatically navigate to the specific move when the page loads.
"""

import json
import os
import csv
from typing import Dict, Any, List

def load_sgf_content(sgf_file: str) -> str:
    """Load SGF content from file."""
    try:
        with open(sgf_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading SGF file {sgf_file}: {e}")
        return ""

def load_analysis_data(analysis_file: str) -> Dict[str, Any]:
    """Load analysis JSON data from file."""
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading analysis file {analysis_file}: {e}")
        return {}

def format_activation_strength(strength: float) -> str:
    """Format activation strength as percentage."""
    return f"{strength:.4f}"

def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value * 100:.1f}"

def generate_channel_bars(channel_activity: List[int]) -> str:
    """Generate HTML for channel activity visualization."""
    if not channel_activity:
        return ""
    
    max_activity = max(channel_activity) if channel_activity else 1
    bars = []
    
    for i, activity in enumerate(channel_activity):
        if max_activity > 0:
            height_percent = (activity / max_activity) * 100
        else:
            height_percent = 0
        
        bar_html = f'''<div class="channel-bar" data-tooltip="Channel {i}: {activity}">
            <div class="channel-fill" style="width: {height_percent}%"></div>
        </div>'''
        bars.append(bar_html)
    
    return '\n'.join(bars)

def generate_policy_moves(policy_moves: List[Dict[str, Any]]) -> str:
    """Generate HTML for top policy moves."""
    if not policy_moves:
        return "<div>No policy moves available</div>"
    
    moves_html = []
    for move in policy_moves:
        coord = move.get('coord', 'Unknown')
        percentage = move.get('percentage', 0)
        count = move.get('count', 0)
        
        move_html = f'''<div class="policy-move">
            <span><strong>{coord}</strong></span>
            <span>{percentage}% ({count})</span>
        </div>'''
        moves_html.append(move_html)
    
    return '\n'.join(moves_html)

def generate_component_activations(activations: List[float]) -> str:
    """Generate HTML for component activation visualization."""
    if not activations:
        return "<div>No activation data available</div>"
    
    activations_html = []
    for i, activation in enumerate(activations):
        percent = activation * 100
        activations_html.append(f'''
        <div style="margin: 5px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>Component {i}:</span>
                <span>{activation:.4f}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percent:.1f}%"></div>
            </div>
        </div>''')
    
    return '\n'.join(activations_html)

def parse_sgf_moves(sgf_content: str, target_turn: int) -> tuple:
    """Parse SGF content and extract stone positions up to target turn.
    
    Since we're now using Besogo which handles SGF directly, we only need to 
    extract the move of interest for highlighting purposes.
    """
    import re
    
    # Extract moves from SGF for move highlighting purposes
    moves = []
    move_pattern = r'[BW]\[[a-z]{0,2}\]'
    
    for match in re.finditer(move_pattern, sgf_content):
        move_text = match.group()
        color = 'B' if move_text[0] == 'B' else 'W'
        coord_text = move_text[2:-1]  # Extract coordinate between brackets
        
        if coord_text == '':  # Pass move
            moves.append((color, None))
        else:
            # Convert SGF coordinates to board position (a=0, b=1, etc.)
            if len(coord_text) == 2:
                sgf_col = ord(coord_text[0]) - ord('a')  # 0-6 for 7x7
                sgf_row = ord(coord_text[1]) - ord('a')  # 0-6 for 7x7
                
                # Direct mapping for 7x7 board - no offset needed
                if 0 <= sgf_col < 7 and 0 <= sgf_row < 7:
                    moves.append((color, (sgf_row, sgf_col)))
                else:
                    moves.append((color, None))  # Outside 7x7 region
            else:
                moves.append((color, None))  # Invalid or pass
    
    # Find move of interest
    move_of_interest = None
    if target_turn < len(moves):
        _, move_of_interest = moves[target_turn]
    
    return {}, move_of_interest, moves


# SVG generation functions removed - now using Besogo

def get_html_template() -> str:
    """Return the embedded HTML template using Besogo."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    
    <!-- Besogo CSS and JS -->
    <link rel="stylesheet" href="besogo/besogo.css">
    <link rel="stylesheet" href="besogo/board-flat.css">
    <script src="besogo/besogo.all.js"></script>

    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2em;
        }
        .header .subtitle {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
        }
        .board-section {
            display: flex;
            flex-direction: column;
        }
        .analysis-section {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .analysis-card {
            background: #f8f9fa;
            border-radius: 6px;
            padding: 15px;
            border-left: 4px solid #667eea;
        }
        .analysis-card h3 {
            margin: 0 0 15px 0;
            color: #2c3e50;
            font-size: 1.2em;
        }
        .data-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }
        .data-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #e9ecef;
        }
        .data-label {
            font-weight: 600;
            color: #495057;
            cursor: help;
        }
        
        .tooltip-icon {
            color: #667eea;
            font-size: 0.8em;
            margin-left: 4px;
            cursor: help;
        }
        
        /* Ensure tooltips are visible */
        [data-tooltip] {
            position: relative;
        }
        
        [data-tooltip]:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #f0f0f0;
            color: #333;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            white-space: normal;
            max-width: 300px;
            width: max-content;
            z-index: 10000;
            pointer-events: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            word-wrap: break-word;
            line-height: 1.4;
            border: 1px solid #ccc;
        }
        .data-value {
            color: #6c757d;
            font-family: 'Courier New', monospace;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        .policy-moves {
            max-height: 200px;
            overflow-y: auto;
        }
        .policy-move {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 10px;
            margin: 2px 0;
            background: white;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }
        .channel-activity {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(30px, 1fr));
            gap: 2px;
            margin: 10px 0;
        }
        .channel-bar {
            height: 20px;
            background: #e9ecef;
            border-radius: 2px;
            position: relative;
            overflow: hidden;
        }
        .channel-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
        }
        .position-highlight {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 15px;
        }
        .current-move-display {
            background: #e3f2fd;
            border: 2px solid #2196f3;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 15px;
            text-align: center;
            font-size: 1.1em;
        }
        .position-navigation {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        .nav-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            text-decoration: none;
            display: inline-block;
            transition: transform 0.2s, opacity 0.2s;
        }
        .nav-button:hover {
            transform: scale(1.05);
            text-decoration: none;
            color: white;
        }
        .nav-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .position-counter {
            font-weight: bold;
            color: #2c3e50;
            font-family: 'Courier New', monospace;
        }
        
        /* Besogo board styling */
        .besogo-viewer {
            width: 100%;
            height: 600px;
            border: 2px solid #8B4513;
            border-radius: 8px;
            overflow: visible !important;
            position: relative;
            z-index: 1000;
        }
        
        /* Fix Besogo control visibility */
        .besogo-viewer * {
            position: relative !important;
            z-index: 1001 !important;
        }
        
        @media (max-width: 1024px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{TITLE}}</h1>
            <div class="subtitle">{{SUBTITLE}}</div>
        </div>
        
        <div class="position-navigation">
            <a href="{{PREV_POS_HTML}}" class="nav-button" {{PREV_DISABLED}} data-tooltip="{{PREV_TITLE}}">← Previous Position</a>
            <span class="position-counter">Position {{CURRENT_INDEX}} of {{TOTAL_POSITIONS}}</span>
            <a href="{{NEXT_POS_HTML}}" class="nav-button" {{NEXT_DISABLED}} data-tooltip="{{NEXT_TITLE}}">Next Position →</a>
        </div>
        
        <div class="content">
            <div class="board-section">
                <div class="position-highlight">
                    <strong>Move of Interest:</strong> {{MOVE_COORD}} at Turn {{TURN_NUMBER}}
                    <br><strong>Activation Strength:</strong> {{ACTIVATION_STRENGTH}}
                </div>
                
                <div class="current-move-display">
                    <strong>Current Move:</strong> <span id="current-move-display">Turn {{TURN_NUMBER}}</span>
                </div>
                
                <!-- Besogo Go Board -->
                <div class="besogo-viewer" 
                     size="7" 
                     coord="western"
                     panels="control+names"
                     orient="portrait"
                     portratio="none"
                     path="{{TURN_NUMBER}}">{{SGF_CONTENT}}</div>
            </div>
            
            <div class="analysis-section">
                <div class="analysis-card">
                    <h3>🧠 NMF Component Analysis</h3>
                    <div class="data-grid">
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Index of the NMF part (group of components) that this activation belongs to. Parts partition the model into sets of interpretable patterns.">Part: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{PART}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Rank of this position within the part, ordered by activation strength (1 = strongest example of this component).">Rank: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{RANK}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Unique identifier for this position in the entire dataset, useful for cross-referencing analyses and SGF files.">Global Position: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{GLOBAL_POS}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Percentile of the activation strength when compared with ALL positions in the dataset (e.g. 99 % means stronger than 99 % of positions).">Activation Percentile: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{ACTIVATION_PERCENTILE}}%</span>
                        </div>
                    </div>
                    
                    <div>
                        <strong data-tooltip="Raw activation value (0-1) output by the model for this component at this move; higher values indicate the pattern is strongly present in the board position.">Activation Strength: {{ACTIVATION_STRENGTH}} <span class="tooltip-icon">ⓘ</span></strong>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {{ACTIVATION_PERCENT}}%"></div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <strong data-tooltip="How many convolutional channels fired above threshold and which ones; gives a low-level view of network attention on the board.">Channel Activity ({{TOTAL_BOARD_ACTIVITY}} active channels) <span class="tooltip-icon">ⓘ</span></strong>
                        <div class="channel-activity">
                            {{CHANNEL_BARS}}
                        </div>
                    </div>
                </div>
                
                <div class="analysis-card">
                    <h3>🎯 Go Pattern Analysis</h3>
                    <div class="data-grid">
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Categorisation of the move (normal play, pass, resign) to understand strategic intent or special game events.">Move Type: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{MOVE_TYPE}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Stage of the game inferred from move number and board state: opening, middle-game or endgame.">Game Phase: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{GAME_PHASE}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Shannon entropy of the model's move probability distribution; low entropy indicates high confidence concentrated on a few moves.">Policy Entropy: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{POLICY_ENTROPY}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Probability assigned by the neural network to the selected move – effectively its confidence in that play.">Policy Confidence: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{POLICY_CONFIDENCE}}%</span>
                        </div>
                    </div>
                    
                    <div>
                        <strong data-tooltip="List of moves the policy network thinks are best, with their probabilities and visit counts; helps explain the AI's tactical choices.">Top Policy Moves: <span class="tooltip-icon">ⓘ</span></strong>
                        <div class="policy-moves">
                            {{POLICY_MOVES}}
                        </div>
                    </div>
                </div>
                
                <div class="analysis-card">
                    <h3>📊 Component Comparison</h3>
                    <div class="data-grid">
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Measure (0-1) of how distinct this component's activation pattern is compared to other components – higher means less overlap.">Uniqueness Score: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{UNIQUENESS_SCORE}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Ordering of components by average activation strength across all positions, where 1 is the most frequently strongest pattern.">Component Rank: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{COMPONENT_RANK}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Highest activation value among ALL other components at this position – used to assess selectivity of the current component.">Max Other Activation: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{MAX_OTHER_ACTIVATION}}</span>
                        </div>
                    </div>
                    
                    <div>
                        <strong data-tooltip="Bar chart of activation values for EVERY component so you can see the full activation profile of this position.">Activation in All Components: <span class="tooltip-icon">ⓘ</span></strong>
                        {{COMPONENT_ACTIVATIONS}}
                    </div>
                </div>
                
                <div class="analysis-card">
                    <h3>📁 File References</h3>
                    <div class="data-item">
                        <span class="data-label" data-tooltip="Original Smart-Game-Format game file from which this position was extracted.">SGF File: <span class="tooltip-icon">ⓘ</span></span>
                        <span class="data-value">{{SGF_FILE}}</span>
                    </div>
                    <div class="data-item">
                        <span class="data-label" data-tooltip="NumPy binary file containing the encoded board tensor used as input to the model.">Board Tensor: <span class="tooltip-icon">ⓘ</span></span>
                        <span class="data-value">{{BOARD_NPY}}</span>
                    </div>
                    <div class="data-item">
                        <span class="data-label" data-tooltip="Compressed KataGo self-play NPZ file that provided raw tensors and move statistics for this position.">NPZ Source: <span class="tooltip-icon">ⓘ</span></span>
                        <span class="data-value">{{NPZ_FILE}}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Initialize Besogo after page load
        document.addEventListener('DOMContentLoaded', function() {
            besogo.autoInit();
            console.log('Besogo Go board initialized for position {{GLOBAL_POS}}');
            
            // Track current move as user navigates through the game
            const currentMoveDisplay = document.getElementById('current-move-display');
            
            // Function to calculate move number from root to current node
            function calculateMoveNumber(editor) {
                if (!editor || !editor.getCurrent || !editor.getRoot) return 0;
                
                const root = editor.getRoot();
                const current = editor.getCurrent();
                if (!root || !current) return 0;
                
                let moveNumber = 0;
                let node = current;
                
                // Count moves from current node back to root
                while (node && node !== root) {
                    if (node.move) {
                        moveNumber++;
                    }
                    node = node.parent;
                }
                
                // Return the actual move number (no subtraction)
                return moveNumber;
            }
            
            // Function to update current move display
            function updateCurrentMove() {
                const besogoViewer = document.querySelector('.besogo-viewer');
                if (!besogoViewer || !besogoViewer.besogoEditor) return;
                
                const editor = besogoViewer.besogoEditor;
                const moveNumber = calculateMoveNumber(editor);
                currentMoveDisplay.textContent = `Turn ${moveNumber}`;
            }
            
            // Wait for Besogo to initialize and then set up the listener
            function setupBesogoListener() {
                const besogoViewer = document.querySelector('.besogo-viewer');
                if (!besogoViewer || !besogoViewer.besogoEditor) {
                    // Try again in a bit if Besogo isn't ready yet
                    setTimeout(setupBesogoListener, 100);
                    return;
                }
                
                const editor = besogoViewer.besogoEditor;
                
                // Add listener for navigation changes
                editor.addListener(function(msg) {
                    if (msg.navChange) {
                        updateCurrentMove();
                    }
                });
                
                // Initial update
                updateCurrentMove();
            }
            
            // Start setup after a short delay to ensure Besogo is initialized
            setTimeout(setupBesogoListener, 500);
        });
    </script>
</body>
</html>'''


def generate_html_file(output_path: str, data: Dict[str, Any]) -> None:
    """Generate HTML file from template and data."""
    try:
        template = get_html_template()
        
        # Replace all template variables
        for key, value in data.items():
            placeholder = f"{{{{{key}}}}}"
            template = template.replace(placeholder, str(value))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template)
        
        print(f"Generated: {output_path}")
        
    except Exception as e:
        print(f"Error generating HTML file {output_path}: {e}")

def process_position(summary_row: Dict[str, str], output_dir: str, all_positions: List[Dict[str, str]]) -> None:
    """Process a single position and generate its HTML file."""
    global_pos = summary_row['global_pos']
    part = summary_row['part']
    rank = summary_row['rank']
    
    # Find current position index and navigation data
    current_index = -1
    for i, pos in enumerate(all_positions):
        if pos['global_pos'] == global_pos:
            current_index = i
            break
    
    # Get previous and next positions
    prev_pos = all_positions[current_index - 1] if current_index > 0 else None
    next_pos = all_positions[current_index + 1] if current_index < len(all_positions) - 1 else None
    
    # Load data files
    sgf_content = load_sgf_content(summary_row['sgf_file'])
    analysis_data = load_analysis_data(summary_row['analysis_file'])
    
    if not analysis_data:
        print(f"Skipping position {global_pos} - no analysis data")
        return
    
    # Extract data for template
    position_info = analysis_data.get('position_info', {})
    nmf_analysis = analysis_data.get('nmf_analysis', {})
    go_pattern = analysis_data.get('go_pattern_analysis', {})
    component_comp = analysis_data.get('component_comparison', {})
    
    # Parse SGF for move information (Besogo handles the board display)
    turn_number = int(position_info.get('turn_number', 0))
    board_state, move_of_interest, all_moves = parse_sgf_moves(sgf_content, turn_number)
    
    # Calculate the correct display turn number (subtract 1 to match expected numbering)
    display_turn_number = max(0, turn_number - 1)
    
    # Prepare template data
    template_data = {
        'TITLE': f"Position {global_pos} Analysis",
        'SUBTITLE': f"Part {part}, Rank {rank} - NMF Component Analysis",
        'CURRENT_INDEX': current_index + 1,
        'TOTAL_POSITIONS': len(all_positions),
        'PREV_POS': prev_pos['global_pos'] if prev_pos else None,
        'NEXT_POS': next_pos['global_pos'] if next_pos else None,
        'PREV_TITLE': f"Position {prev_pos['global_pos']} (Part {prev_pos['part']}, Rank {prev_pos['rank']})" if prev_pos else None,
        'NEXT_TITLE': f"Position {next_pos['global_pos']} (Part {next_pos['part']}, Rank {next_pos['rank']})" if next_pos else None,
        'PREV_POS_HTML': f"pos_{prev_pos['global_pos']}_analysis.html" if prev_pos else "#",
        'NEXT_POS_HTML': f"pos_{next_pos['global_pos']}_analysis.html" if next_pos else "#",
        'PREV_DISABLED': 'style="opacity: 0.5; pointer-events: none;"' if not prev_pos else '',
        'NEXT_DISABLED': 'style="opacity: 0.5; pointer-events: none;"' if not next_pos else '',
        'PART': part,
        'RANK': rank,
        'GLOBAL_POS': global_pos,
        'TURN_NUMBER': str(display_turn_number),
        'MOVE_COORD': position_info.get('move_coordinate', 'Unknown'),
        'SGF_CONTENT': sgf_content,  # Raw SGF for Besogo
        'SGF_FILE': summary_row['sgf_file'],
        'BOARD_NPY': summary_row['board_npy'],
        'NPZ_FILE': position_info.get('npz_file', 'Unknown'),
        
        # NMF Analysis
        'ACTIVATION_STRENGTH': format_activation_strength(nmf_analysis.get('activation_strength', 0)),
        'ACTIVATION_PERCENT': format_percentage(nmf_analysis.get('activation_strength', 0)),
        'ACTIVATION_PERCENTILE': f"{component_comp.get('activation_percentile', 0):.2f}",
        'TOTAL_BOARD_ACTIVITY': nmf_analysis.get('total_board_activity', 0),
        'CHANNEL_BARS': generate_channel_bars(nmf_analysis.get('channel_activity', [])),
        
        # Go Pattern Analysis
        'MOVE_TYPE': go_pattern.get('move_type', 'Unknown').title(),
        'GAME_PHASE': go_pattern.get('game_phase', 'Unknown').replace('_', ' ').title(),
        'POLICY_ENTROPY': f"{go_pattern.get('policy_entropy', 0):.3f}",
        'POLICY_CONFIDENCE': go_pattern.get('policy_confidence', 0),
        'POLICY_MOVES': generate_policy_moves(go_pattern.get('top_policy_moves', [])),
        
        # Component Comparison
        'UNIQUENESS_SCORE': f"{component_comp.get('uniqueness_score', 0):.4f}",
        'COMPONENT_RANK': component_comp.get('component_rank', 'Unknown'),
        'MAX_OTHER_ACTIVATION': f"{component_comp.get('max_other_component_activation', 0):.4f}",
        'COMPONENT_ACTIVATIONS': generate_component_activations(
            nmf_analysis.get('activation_in_other_components', [])
        )
    }
    
    # Generate output filename
    output_filename = f"pos_{global_pos}_analysis.html"
    output_path = os.path.join(output_dir, output_filename)
    
    # Generate HTML file
    generate_html_file(output_path, template_data)

def generate_index_page(summary_data: List[Dict[str, str]], output_dir: str) -> None:
    """Generate index page linking to all position analyses."""
    index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Go Position Analysis - Step 5 Results</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #667eea;
        }
        .header h1 {
            color: #2c3e50;
            margin: 0 0 10px 0;
        }
        .header p {
            color: #6c757d;
            font-size: 1.1em;
        }
        .positions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        .position-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .position-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .position-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .position-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }
        .detail-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #dee2e6;
        }
        .detail-label {
            font-weight: 600;
            color: #495057;
        }
        .detail-value {
            color: #6c757d;
            font-family: 'Courier New', monospace;
        }
        .view-button {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        .view-button:hover {
            transform: scale(1.05);
            text-decoration: none;
            color: white;
        }
        .part-section {
            margin-bottom: 40px;
        }
        .part-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .part-header h2 {
            margin: 0;
            font-size: 1.5em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Go Position Analysis Results</h1>
            <p>Step 5: NMF Component Analysis - Positions of Interest</p>
            <p>9 positions analyzed across 3 NMF parts, each showing the top 3 strongest activations</p>
        </div>
        
'''
    
    # Group positions by part
    parts = {}
    for row in summary_data:
        part = row['part']
        if part not in parts:
            parts[part] = []
        parts[part].append(row)
    
    # Generate sections for each part
    for part_num in sorted(parts.keys()):
        positions = parts[part_num]
        index_html += f'''
        <div class="part-section">
            <div class="part-header">
                <h2>NMF Part {part_num}</h2>
            </div>
            <div class="positions-grid">
'''
        
        # Sort by rank
        positions.sort(key=lambda x: int(x['rank']))
        
        for pos in positions:
            index_html += f'''
                <div class="position-card">
                    <div class="position-title">Position {pos['global_pos']}</div>
                    <div class="position-details">
                        <div class="detail-item">
                            <span class="detail-label">Rank:</span>
                            <span class="detail-value">{pos['rank']}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Move:</span>
                            <span class="detail-value">{pos['coord']}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Turn:</span>
                            <span class="detail-value">{pos['turn']}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Part:</span>
                            <span class="detail-value">{pos['part']}</span>
                        </div>
                    </div>
                    <a href="pos_{pos['global_pos']}_analysis.html" class="view-button">
                        View Analysis →
                    </a>
                </div>
'''
        
        index_html += '''
            </div>
        </div>
'''
    
    index_html += '''
    </div>
</body>
</html>
'''
    
    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    print(f"Generated index page: {index_path}")

def main():
    """Main function to generate all HTML reports."""
    # Set up paths
    base_dir = os.path.dirname(__file__)
    summary_file = os.path.join(base_dir, 'strong_positions_summary.csv')
    output_dir = os.path.join(base_dir, 'html_reports')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load summary data
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            summary_data = list(reader)
    except Exception as e:
        print(f"Error loading summary file: {e}")
        return
    
    print(f"Processing {len(summary_data)} positions...")
    
    # Process each position
    os.chdir(base_dir)  # Change to base directory for relative file paths
    
    for row in summary_data:
        process_position(row, output_dir, summary_data)
    
    # Generate index page
    generate_index_page(summary_data, output_dir)
    
    print(f"\nAll HTML reports generated in: {output_dir}")
    print("Open index.html to view all analyses")

if __name__ == "__main__":
    main() 