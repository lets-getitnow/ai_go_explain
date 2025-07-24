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
        
        bar_html = f'''<div class="channel-bar" title="Channel {i}: {activity}">
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
    
    SGF contains 9x9 coordinates, but analysis was done on 7x7.
    Map 9x9 SGF coordinates to 7x7 analysis coordinates by taking center region.
    """
    import re
    
    # Extract moves from SGF
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
    
    # Build board state up to target turn
    board = {}  # {(row, col): 'B' or 'W'}
    move_of_interest = None
    
    for i, (color, pos) in enumerate(moves):
        if i >= target_turn:
            if i == target_turn and pos:
                move_of_interest = pos
            break
        if pos:
            board[pos] = color
    
    return board, move_of_interest, moves

def generate_grid_lines() -> str:
    """Generate SVG grid lines for 7x7 Go board."""
    lines = []
    
    # Horizontal lines
    for i in range(7):
        y = 60 + i * 40
        lines.append(f'<line x1="60" y1="{y}" x2="300" y2="{y}" stroke="#8B4513" stroke-width="1"/>')
    
    # Vertical lines  
    for i in range(7):
        x = 60 + i * 40
        lines.append(f'<line x1="{x}" y1="60" x2="{x}" y2="300" stroke="#8B4513" stroke-width="1"/>')
    
    # Star points (handicap points) for 7x7
    star_points = [(1, 1), (1, 5), (5, 1), (5, 5), (3, 3)]
    for row, col in star_points:
        x = 60 + col * 40
        y = 60 + row * 40
        lines.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#8B4513"/>')
    
    return '\n'.join(lines)

def generate_coord_labels() -> str:
    """Generate coordinate labels for 7x7 board."""
    labels = []
    
    # Column labels (A-G for 7x7)
    cols = 'ABCDEFG'
    for i, letter in enumerate(cols):
        x = 60 + i * 40
        labels.append(f'<text x="{x}" y="50" text-anchor="middle" font-family="Arial" font-size="12" fill="#8B4513">{letter}</text>')
        labels.append(f'<text x="{x}" y="320" text-anchor="middle" font-family="Arial" font-size="12" fill="#8B4513">{letter}</text>')
    
    # Row labels (1-7)
    for i in range(7):
        y = 60 + i * 40 + 4  # +4 for text baseline
        row_num = 7 - i  # Go coordinates start from bottom
        labels.append(f'<text x="45" y="{y}" text-anchor="middle" font-family="Arial" font-size="12" fill="#8B4513">{row_num}</text>')
        labels.append(f'<text x="315" y="{y}" text-anchor="middle" font-family="Arial" font-size="12" fill="#8B4513">{row_num}</text>')
    
    return '\n'.join(labels)

def generate_stones(board: dict) -> str:
    """Generate SVG stones from board position."""
    stones = []
    
    for (row, col), color in board.items():
        if 0 <= row < 7 and 0 <= col < 7:
            x = 60 + col * 40
            y = 60 + row * 40
            
            if color == 'B':
                stones.append(f'<circle cx="{x}" cy="{y}" r="16" fill="url(#blackStone)" stroke="#000" stroke-width="1"/>')
            else:  # White
                stones.append(f'<circle cx="{x}" cy="{y}" r="16" fill="url(#whiteStone)" stroke="#666" stroke-width="1"/>')
    
    return '\n'.join(stones)

def generate_move_marker(move_of_interest: tuple, move_coord: str) -> str:
    """Generate marker for the move of interest."""
    if not move_of_interest:
        return ""
    
    row, col = move_of_interest
    if 0 <= row < 7 and 0 <= col < 7:
        x = 60 + col * 40
        y = 60 + row * 40
        
        return f'''
        <circle cx="{x}" cy="{y}" r="20" fill="none" stroke="#ff0000" stroke-width="3" opacity="0.8"/>
        <text x="{x}" y="{y-25}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#ff0000">← Move of Interest</text>
        '''
    return ""

def generate_moves_javascript(moves: list) -> str:
    """Convert moves list to JavaScript array format."""
    js_moves = []
    
    for color, pos in moves:
        if pos:
            js_move = f'{{"color": "{color}", "pos": [{pos[0]}, {pos[1]}]}}'
        else:
            js_move = f'{{"color": "{color}", "pos": null}}'
        js_moves.append(js_move)
    
    return '[' + ', '.join(js_moves) + ']'

def get_html_template() -> str:
    """Return the embedded HTML template."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>

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
        .go-board-container {
            border: 2px solid #8B4513;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
            background: #f4f4f4;
            padding: 10px;
        }
        .go-board-svg {
            margin: 0 auto;
            display: block;
            border-radius: 4px;
        }
        .board-controls {
            margin-top: 15px;
        }
        .control-buttons {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-bottom: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }
        .control-buttons button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s, opacity 0.2s;
        }
        .control-buttons button:hover {
            transform: scale(1.1);
        }
        .control-buttons button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .move-display {
            font-weight: bold;
            color: #2c3e50;
            margin: 0 10px;
            font-family: 'Courier New', monospace;
        }
        .position-info {
            text-align: center;
            font-size: 0.9em;
            color: #666;
        }
        .position-info a {
            color: #667eea;
            text-decoration: none;
        }
        .position-info a:hover {
            text-decoration: underline;
        }
        .position-highlight {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 15px;
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
            <a href="{{PREV_POS_HTML}}" class="nav-button" {{PREV_DISABLED}} title="{{PREV_TITLE}}">← Previous Position</a>
            <span class="position-counter">Position {{CURRENT_INDEX}} of {{TOTAL_POSITIONS}}</span>
            <a href="{{NEXT_POS_HTML}}" class="nav-button" {{NEXT_DISABLED}} title="{{NEXT_TITLE}}">Next Position →</a>
        </div>
        
        <div class="content">
            <div class="board-section">
                <div class="position-highlight">
                    <strong>Move of Interest:</strong> {{MOVE_COORD}} at Turn {{TURN_NUMBER}}
                    <br><strong>Activation Strength:</strong> {{ACTIVATION_STRENGTH}}
                </div>
                
                <div id="go-board-{{GLOBAL_POS}}" class="go-board-container">
                    <svg width="360" height="360" viewBox="0 0 360 360" class="go-board-svg">
                        <!-- Board grid -->
                        <defs>
                            <pattern id="wood" patternUnits="userSpaceOnUse" width="40" height="40">
                                <rect width="40" height="40" fill="#DEB887"/>
                                <rect width="40" height="40" fill="url(#woodGrain)" opacity="0.3"/>
                            </pattern>
                            <linearGradient id="woodGrain" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#CD853F;stop-opacity:1" />
                                <stop offset="100%" style="stop-color:#F4A460;stop-opacity:1" />
                            </linearGradient>
                            <!-- Black stone gradient -->
                            <radialGradient id="blackStone" cx="30%" cy="30%">
                                <stop offset="0%" stop-color="#555"/>
                                <stop offset="70%" stop-color="#222"/>
                                <stop offset="100%" stop-color="#000"/>
                            </radialGradient>
                            <!-- White stone gradient -->
                            <radialGradient id="whiteStone" cx="30%" cy="30%">
                                <stop offset="0%" stop-color="#fff"/>
                                <stop offset="70%" stop-color="#eee"/>
                                <stop offset="100%" stop-color="#ccc"/>
                            </radialGradient>
                        </defs>
                        
                        <!-- Board background -->
                        <rect x="20" y="20" width="320" height="320" fill="url(#wood)" stroke="#8B4513" stroke-width="2"/>
                        
                        <!-- Grid lines -->
                        {{GRID_LINES}}
                        
                        <!-- Coordinate labels -->
                        {{COORD_LABELS}}
                        
                        <!-- Stones -->
                        {{STONES}}
                        
                        <!-- Move of interest marker -->
                        {{MOVE_MARKER}}
                    </svg>
                    
                    <div class="board-controls">
                        <div class="control-buttons">
                            <button onclick="goToMove(0)" title="Go to start">⏮</button>
                            <button onclick="previousMove()" title="Previous move">⏪</button>
                            <span class="move-display">Move: <span id="current-move">{{TURN_NUMBER}}</span> / <span id="total-moves">{{TOTAL_MOVES}}</span></span>
                            <button onclick="nextMove()" title="Next move">⏩</button>
                            <button onclick="goToMove(-1)" title="Go to end">⏭</button>
                        </div>
                        <div class="position-info">
                            <p><strong>Move of Interest: Turn {{TURN_NUMBER}}</strong> | <a href="#" onclick="goToMove({{TURN_NUMBER}}); return false;">Jump to Move of Interest</a></p>
                            <p>SGF: <a href="#" onclick="alert('{{SGF_CONTENT}}'); return false;">View SGF</a></p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="analysis-section">
                <div class="analysis-card">
                    <h3>🧠 NMF Component Analysis</h3>
                    <div class="data-grid">
                        <div class="data-item">
                            <span class="data-label">Part:</span>
                            <span class="data-value">{{PART}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label">Rank:</span>
                            <span class="data-value">{{RANK}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label">Global Position:</span>
                            <span class="data-value">{{GLOBAL_POS}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label">Activation Percentile:</span>
                            <span class="data-value">{{ACTIVATION_PERCENTILE}}%</span>
                        </div>
                    </div>
                    
                    <div>
                        <strong>Activation Strength: {{ACTIVATION_STRENGTH}}</strong>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {{ACTIVATION_PERCENT}}%"></div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <strong>Channel Activity ({{TOTAL_BOARD_ACTIVITY}} active channels)</strong>
                        <div class="channel-activity">
                            {{CHANNEL_BARS}}
                        </div>
                    </div>
                </div>
                
                <div class="analysis-card">
                    <h3>🎯 Go Pattern Analysis</h3>
                    <div class="data-grid">
                        <div class="data-item">
                            <span class="data-label">Move Type:</span>
                            <span class="data-value">{{MOVE_TYPE}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label">Game Phase:</span>
                            <span class="data-value">{{GAME_PHASE}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label">Policy Entropy:</span>
                            <span class="data-value">{{POLICY_ENTROPY}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label">Policy Confidence:</span>
                            <span class="data-value">{{POLICY_CONFIDENCE}}%</span>
                        </div>
                    </div>
                    
                    <div>
                        <strong>Top Policy Moves:</strong>
                        <div class="policy-moves">
                            {{POLICY_MOVES}}
                        </div>
                    </div>
                </div>
                
                <div class="analysis-card">
                    <h3>📊 Component Comparison</h3>
                    <div class="data-grid">
                        <div class="data-item">
                            <span class="data-label">Uniqueness Score:</span>
                            <span class="data-value">{{UNIQUENESS_SCORE}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label">Component Rank:</span>
                            <span class="data-value">{{COMPONENT_RANK}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label">Max Other Activation:</span>
                            <span class="data-value">{{MAX_OTHER_ACTIVATION}}</span>
                        </div>
                    </div>
                    
                    <div>
                        <strong>Activation in All Components:</strong>
                        {{COMPONENT_ACTIVATIONS}}
                    </div>
                </div>
                
                <div class="analysis-card">
                    <h3>📁 File References</h3>
                    <div class="data-item">
                        <span class="data-label">SGF File:</span>
                        <span class="data-value">{{SGF_FILE}}</span>
                    </div>
                    <div class="data-item">
                        <span class="data-label">Board Tensor:</span>
                        <span class="data-value">{{BOARD_NPY}}</span>
                    </div>
                    <div class="data-item">
                        <span class="data-label">NPZ Source:</span>
                        <span class="data-value">{{NPZ_FILE}}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Game state and moves
        let allMoves = {{ALL_MOVES_JS}};
        let currentMoveIndex = {{TURN_NUMBER}};
        let moveOfInterest = {{TURN_NUMBER}};
        
        function updateBoard(moveIndex) {
            // Clear all stones
            const svg = document.querySelector('.go-board-svg');
            const existingStones = svg.querySelectorAll('circle[r="16"]');
            existingStones.forEach(stone => stone.remove());
            
            // Clear move marker
            const existingMarkers = svg.querySelectorAll('circle[r="20"]');
            existingMarkers.forEach(marker => marker.remove());
            const existingTexts = svg.querySelectorAll('text[font-weight="bold"]');
            existingTexts.forEach(text => text.remove());
            
            // Play moves up to current index
            const board = {};
            for (let i = 0; i <= moveIndex && i < allMoves.length; i++) {
                const move = allMoves[i];
                if (move.pos) {
                    board[move.pos[0] + ',' + move.pos[1]] = move.color;
                }
            }
            
            // Draw stones
            for (const [posKey, color] of Object.entries(board)) {
                const [row, col] = posKey.split(',').map(Number);
                const x = 60 + col * 40;
                const y = 60 + row * 40;
                
                const stone = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                stone.setAttribute('cx', x);
                stone.setAttribute('cy', y);
                stone.setAttribute('r', '16');
                stone.setAttribute('fill', color === 'B' ? 'url(#blackStone)' : 'url(#whiteStone)');
                stone.setAttribute('stroke', color === 'B' ? '#000' : '#666');
                stone.setAttribute('stroke-width', '1');
                svg.appendChild(stone);
            }
            
            // Add move marker if this is the move of interest
            if (moveIndex === moveOfInterest && moveIndex < allMoves.length) {
                const move = allMoves[moveIndex];
                if (move.pos) {
                    const x = 60 + move.pos[1] * 40;
                    const y = 60 + move.pos[0] * 40;
                    
                    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    marker.setAttribute('cx', x);
                    marker.setAttribute('cy', y);
                    marker.setAttribute('r', '20');
                    marker.setAttribute('fill', 'none');
                    marker.setAttribute('stroke', '#ff0000');
                    marker.setAttribute('stroke-width', '3');
                    marker.setAttribute('opacity', '0.8');
                    svg.appendChild(marker);
                    
                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', x);
                    text.setAttribute('y', y - 25);
                    text.setAttribute('text-anchor', 'middle');
                    text.setAttribute('font-family', 'Arial');
                    text.setAttribute('font-size', '12');
                    text.setAttribute('font-weight', 'bold');
                    text.setAttribute('fill', '#ff0000');
                    text.textContent = '← Move of Interest';
                    svg.appendChild(text);
                }
            }
            
            // Update UI
            currentMoveIndex = moveIndex;
            document.getElementById('current-move').textContent = moveIndex;
            
            // Update button states
            const buttons = document.querySelectorAll('.control-buttons button');
            buttons[0].disabled = moveIndex <= 0; // First
            buttons[1].disabled = moveIndex <= 0; // Previous
            buttons[3].disabled = moveIndex >= allMoves.length - 1; // Next
            buttons[4].disabled = moveIndex >= allMoves.length - 1; // Last
        }
        
        function goToMove(moveIndex) {
            if (moveIndex === -1) moveIndex = allMoves.length - 1;
            moveIndex = Math.max(0, Math.min(allMoves.length - 1, moveIndex));
            updateBoard(moveIndex);
        }
        
        function previousMove() {
            if (currentMoveIndex > 0) {
                updateBoard(currentMoveIndex - 1);
            }
        }
        
        function nextMove() {
            if (currentMoveIndex < allMoves.length - 1) {
                updateBoard(currentMoveIndex + 1);
            }
        }
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Go board visualization loaded with', allMoves.length, 'moves');
            updateBoard(currentMoveIndex);
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
    
    # Parse SGF and generate board
    turn_number = int(position_info.get('turn_number', 0))
    board_state, move_of_interest, all_moves = parse_sgf_moves(sgf_content, turn_number)
    
    # Generate board state up to target turn for initial display
    initial_board = {}
    for i, (color, pos) in enumerate(all_moves):
        if i >= turn_number:
            break
        if pos:
            initial_board[pos] = color
    
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
        'TURN_NUMBER': position_info.get('turn_number', '0'),
        'MOVE_COORD': position_info.get('move_coordinate', 'Unknown'),
        'SGF_CONTENT': sgf_content.replace('\n', '').replace('"', '&quot;'),
        'SGF_FILE': summary_row['sgf_file'],
        'BOARD_NPY': summary_row['board_npy'],
        'NPZ_FILE': position_info.get('npz_file', 'Unknown'),
        
        # Board elements (initial state)
        'GRID_LINES': generate_grid_lines(),
        'COORD_LABELS': generate_coord_labels(),
        'STONES': generate_stones(initial_board),
        'MOVE_MARKER': generate_move_marker(move_of_interest, position_info.get('move_coordinate', 'Unknown')),
        
        # JavaScript game data
        'ALL_MOVES_JS': generate_moves_javascript(all_moves),
        'TOTAL_MOVES': len(all_moves) - 1,
        
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