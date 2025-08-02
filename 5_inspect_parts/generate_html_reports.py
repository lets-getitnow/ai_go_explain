#!/usr/bin/env python3
"""
Generate HTML visualization reports for each position of interest from step 5.

This script creates an HTML file for each analyzed position, displaying:
- Prominent GitHub repository link with logo at the top of each page
- Interactive Go board automatically positioned at the move of interest
- Complete NMF analysis data
- Go pattern analysis with policy moves in Go coordinate format (e.g., C3, D4)
- Part comparison data

Requirements: 
- Each page must have a prominent GitHub link to https://github.com/lets-getitnow/ai_go_explain with GitHub logo
- The Go board must automatically navigate to the specific move when the page loads.
- Policy moves must be displayed in standard Go coordinate format (A-G, 1-7 for 7x7 board).
- Tuple coordinates from analysis data (e.g., "(2,4)") must be converted to Go coordinates (e.g., "C3").
- All CSS styling must be in external CSS files, not inline styles.
"""

import json
import os
import csv
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

def load_sgf_content(sgf_file: str) -> str:
    """Load SGF content from file."""
    try:
        # Handle both old format (direct file) and new format (in output directory)
        if os.path.exists(sgf_file):
            # Direct file path
            with open(sgf_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            # New structured format: sgf_file is like "pos_4683/game.sgf"
            output_path = os.path.join("output", sgf_file)
            with open(output_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception as e:
        print(f"Error loading SGF file {sgf_file}: {e}")
        return ""

def load_analysis_data(analysis_file: str) -> Dict[str, Any]:
    """Load analysis JSON data from file."""
    try:
        # Handle both old format (direct file) and new format (in output directory)
        if os.path.exists(analysis_file):
            # Direct file path
            with open(analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # New structured format: analysis_file is like "pos_4683/analysis.json"
            output_path = os.path.join("output", analysis_file)
            with open(output_path, 'r', encoding='utf-8') as f:
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

def convert_tuple_to_go_coord(tuple_coord: str, board_size: int = 13) -> str:
    """Convert tuple coordinate format to Go coordinate format.
    
    Args:
        tuple_coord: Coordinate in format "(row,col)" or "PASS"
        board_size: Board size (default 13 for human games)
        
    Returns:
        Go coordinate in format "A1" through "M13" for 13x13 board, or "PASS"
    """
    if tuple_coord == "PASS":
        return "PASS"
    
    # Handle tuple format like "(2,4)"
    if tuple_coord.startswith("(") and tuple_coord.endswith(")"):
        try:
            # Extract row and col from "(row,col)"
            coord_str = tuple_coord[1:-1]  # Remove parentheses
            row, col = map(int, coord_str.split(","))
            
            # Convert to Go coordinates (A-M for columns, 1-13 for rows for 13x13)
            # Note: tuple coordinates are 0-indexed, Go coordinates are 1-indexed
            go_col = chr(ord('A') + col)  # A=0, B=1, C=2, etc.
            go_row = str(row + 1)  # 0->1, 1->2, etc.
            
            return f"{go_col}{go_row}"
        except (ValueError, IndexError):
            # If conversion fails, return original
            return tuple_coord
    
    # If already in Go format or unknown format, return as is
    return tuple_coord

def generate_channel_bars(channel_activity: List[Dict[str, Any]]) -> str:
    """Generate HTML for channel activity bars."""
    if not channel_activity:
        return "<div>No channel activity data available</div>"
    
    bars_html = []
    for channel_info in channel_activity:
        if isinstance(channel_info, dict):
            # New format from enhanced analysis
            channel_num = channel_info.get('channel', 0)
            activity = channel_info.get('activity', 0)
            max_region = channel_info.get('max_region', 0)
            min_region = channel_info.get('min_region', 0)
            
            # Convert activity to percentage (assuming max activity is around 1.0)
            percentage = min(100, activity * 100)
            
            bar_html = f'''<div class="channel-bar">
                <span>Channel {channel_num}:</span>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {percentage:.1f}%"></div>
                </div>
                <span>{activity:.4f}</span>
            </div>'''
        else:
            # Old format from selfplay analysis
            channel_num = channel_info
            bar_html = f'''<div class="channel-bar">
                <span>Channel {channel_num}:</span>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 100%"></div>
                </div>
                <span>Active</span>
            </div>'''
        
        bars_html.append(bar_html)
    
    return '\n'.join(bars_html)

def generate_policy_moves(policy_moves: List[Dict[str, Any]]) -> str:
    """Generate HTML for top policy moves with Go coordinate format."""
    if not policy_moves:
        return "<div>No policy moves available</div>"
    
    moves_html = []
    for move in policy_moves:
        # Handle both old format (coord, percentage, count) and new format (move, probability, logit)
        if 'move' in move:
            # New format from enhanced analysis
            coord = move.get('move', 'Unknown')
            percentage = move.get('probability', 0) * 100  # Convert to percentage
            count = 0  # Not available in new format
        else:
            # Old format from selfplay analysis
            coord = move.get('coord', 'Unknown')
            percentage = move.get('percentage', 0)
            count = move.get('count', 0)
        
        # Convert tuple coordinates to Go coordinates if needed
        if isinstance(coord, str) and coord != 'Unknown' and coord != 'PASS':
            go_coord = convert_tuple_to_go_coord(coord)
        else:
            go_coord = coord
        
        move_html = f'''<div class="policy-move">
            <span><strong>{go_coord}</strong></span>
            <span>{percentage:.2f}% ({count})</span>
        </div>'''
        moves_html.append(move_html)
    
    return '\n'.join(moves_html)

def generate_part_activations(activations: List[Dict[str, Any]], all_positions: List[Dict[str, str]] = None) -> str:
    """Generate HTML for part activation visualization with clickable bars."""
    if not activations:
        return "<div>No activation data available</div>"
    
    # Create a mapping of part to its strongest position
    part_to_position = {}
    if all_positions:
        for pos in all_positions:
            # Handle both formats
            if 'part' in pos:
                part = int(pos['part'])
            else:
                part = int(pos.get('part_idx', 0))
            if part not in part_to_position:
                part_to_position[part] = pos
    
    activations_html = []
    for i, activation_info in enumerate(activations):
        if isinstance(activation_info, dict):
            # New format from enhanced analysis
            part_num = activation_info.get('part', i)
            activation = activation_info.get('activation', 0)
        else:
            # Old format from selfplay analysis
            part_num = i
            activation = activation_info
        
        percent = activation * 100
        
        # Create clickable link if we have position data for this part
        if part_num in part_to_position:
            pos = part_to_position[part_num]
            # Handle both formats for position data
            if 'global_pos' in pos:
                global_pos = pos['global_pos']
            else:
                global_pos = pos.get('position_idx', 'N/A')
            
            if 'part' in pos:
                part = pos['part']
            else:
                part = pos.get('part_idx', 'N/A')
            
            if 'rank' in pos:
                rank = pos['rank']
            else:
                rank = '1'
            
            link_url = f"pos_{global_pos}_part{part}_rank{rank}_analysis.html"
            part_html = f'''
        <div style="margin: 5px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <a href="{link_url}" style="text-decoration: none; color: inherit; cursor: pointer;" 
                   title="View strongest example of Part {part_num} (Position {global_pos}, Rank {rank})">
                    <span>Part {part_num}:</span>
                </a>
                <span>{activation:.4f}</span>
            </div>
            <a href="{link_url}" style="text-decoration: none; color: inherit; cursor: pointer;" 
               title="View strongest example of Part {part_num} (Position {global_pos}, Rank {rank})">
                <div class="progress-bar" style="cursor: pointer;">
                    <div class="progress-fill" style="width: {percent:.1f}%"></div>
                </div>
            </a>
        </div>'''
        else:
            # Non-clickable version for parts without position data
            part_html = f'''
        <div style="margin: 5px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>Part {part_num}:</span>
                <span>{activation:.4f}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percent:.1f}%"></div>
            </div>
        </div>'''
        
        activations_html.append(part_html)
    
    return '\n'.join(activations_html)

def parse_sgf_moves(sgf_content: str, target_turn: int, board_size: int = 13) -> tuple:
    """Parse SGF moves and find the move of interest.
    
    Args:
        sgf_content: SGF file content
        target_turn: Turn number to highlight (0-indexed)
        board_size: Board size (default 13 for human games)
        
    Returns:
        Tuple of (moves_dict, move_of_interest, moves_list)
    """
    import re
    
    moves = []
    
    # Extract moves from SGF content
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
                sgf_col = ord(coord_text[0]) - ord('a')  # 0-12 for 13x13
                sgf_row = ord(coord_text[1]) - ord('a')  # 0-12 for 13x13
                
                # Direct mapping for 13x13 board - no offset needed
                if 0 <= sgf_col < board_size and 0 <= sgf_row < board_size:
                    moves.append((color, (sgf_row, sgf_col)))
                else:
                    moves.append((color, None))  # Outside board region
            else:
                moves.append((color, None))  # Invalid or pass
    
    # Find move of interest
    move_of_interest = None
    if target_turn < len(moves):
        _, move_of_interest = moves[target_turn]
    
    return {}, move_of_interest, moves


# SVG generation functions removed - now using Besogo

def get_html_template(board_size: int = 13) -> str:
    """Return the embedded HTML template using Besogo."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{{{TITLE}}}}</title>
    
    <!-- Besogo CSS and JS -->
    <link rel="stylesheet" href="besogo/besogo.css">
    <link rel="stylesheet" href="besogo/board-flat.css">
    <link rel="stylesheet" href="besogo/analysis-styles.css">
    <script src="besogo/besogo.all.js"></script>
</head>
<body>
    <div class="github-header">
        <a href="https://github.com/lets-getitnow/ai_go_explain" target="_blank" rel="noopener noreferrer">
            <svg class="github-logo" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            <span class="project-name">ai_go_explain</span>
            <span>View on GitHub</span>
        </a>
    </div>
    <div class="container">
        <div class="header">
            <h1>{{{{TITLE}}}}</h1>
            <div class="subtitle">{{{{SUBTITLE}}}}</div>
            <div class="timestamp" style="color: #ffffff; font-size: 0.9em; margin-top: 5px; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">Generated: {{{{TIMESTAMP}}}}</div>
        </div>
        
        <div class="position-navigation">
            <a href="{{{{FIRST_POS_HTML}}}}" class="nav-button" {{{{FIRST_DISABLED}}}} data-tooltip="{{{{FIRST_TITLE}}}}">⏮ First Position</a>
            <a href="{{{{PREV_POS_HTML}}}}" class="nav-button" {{{{PREV_DISABLED}}}} data-tooltip="{{{{PREV_TITLE}}}}">← Previous Position</a>
            <span class="position-counter">Position {{{{CURRENT_INDEX}}}} of {{{{TOTAL_POSITIONS}}}}</span>
            <a href="{{{{NEXT_POS_HTML}}}}" class="nav-button" {{{{NEXT_DISABLED}}}} data-tooltip="{{{{NEXT_TITLE}}}}">Next Position →</a>
            <a href="{{{{LAST_POS_HTML}}}}" class="nav-button" {{{{LAST_DISABLED}}}} data-tooltip="{{{{LAST_TITLE}}}}">Last Position ⏭</a>
            <span class="keyboard-hint" style="font-size: 0.8em; color: #666; margin-left: 10px;">(Use ← → arrow keys)</span>
        </div>
        
        <div class="content">
            <div class="board-section">
                <div class="position-highlight">
                    <strong>Move of Interest:</strong> {{{{MOVE_COORD}}}} at Turn {{{{TURN_NUMBER}}}}
                    <br><strong>Activation Strength:</strong> {{{{ACTIVATION_STRENGTH}}}}
                </div>
                
                <div class="current-move-display">
                    <strong>Current Move:</strong> <span id="current-move-display">Turn {{{{TURN_NUMBER}}}}</span>
                </div>
                
                <!-- Besogo Go Board -->
                <div class="besogo-viewer" 
                     size="{board_size}" 
                     coord="western"
                     panels="control+names"
                     orient="portrait"
                     portratio="none">{{{{SGF_CONTENT}}}}</div>
            </div>
            
            <div class="analysis-section">
                <div class="analysis-card">
                    <h3>🧠 NMF Part Analysis</h3>
                    <div class="data-grid">
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Index of the NMF part that this activation belongs to. Parts partition the model into sets of interpretable patterns.">Part: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{{{PART}}}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Rank of this position within the part, ordered by activation strength (1 = strongest example of this component).">Rank: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{{{RANK}}}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Unique identifier for this position in the entire dataset, useful for cross-referencing analyses and SGF files.">Global Position: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{{{GLOBAL_POS}}}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Percentile of the activation strength when compared with ALL positions in the dataset (e.g. 99 % means stronger than 99 % of positions).">Activation Percentile: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{{{ACTIVATION_PERCENTILE}}}}%</span>
                        </div>
                    </div>
                    
                    <div>
                        <strong data-tooltip="Raw activation value (0-1) output by the model for this component at this move; higher values indicate the pattern is strongly present in the board position.">Activation Strength: {{{{ACTIVATION_STRENGTH}}}} <span class="tooltip-icon">ⓘ</span></strong>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {{{{ACTIVATION_PERCENT}}}}%"></div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <strong data-tooltip="How many convolutional channels fired above threshold and which ones; gives a low-level view of network attention on the board.">Channel Activity ({{{{TOTAL_BOARD_ACTIVITY}}}} active channels) <span class="tooltip-icon">ⓘ</span></strong>
                        <div class="channel-activity">
                            {{{{CHANNEL_BARS}}}}
                        </div>
                    </div>
                </div>
                
                <div class="analysis-card">
                    <h3>🎯 Go Pattern Analysis</h3>
                    <div class="data-grid">
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Classification of the move type (normal, ko, ladder, etc.).">Move Type: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{{{MOVE_TYPE}}}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Game phase when this position occurred (opening, middle_game, endgame).">Game Phase: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{{{GAME_PHASE}}}}</span>
                        </div>
                    </div>
                </div>
                
                <div class="analysis-card">
                    <h3>📊 Part Comparison</h3>
                    <div class="data-grid">
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Measure (0-1) of how distinct this part's activation pattern is compared to other parts – higher means less overlap.">Uniqueness Score: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{{{UNIQUENESS_SCORE}}}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Ordering of parts by average activation strength across all positions, where 1 is the most frequently strongest pattern.">Part Rank: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{{{PART_RANK}}}}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label" data-tooltip="Highest activation value among ALL other parts at this position – used to assess selectivity of the current part.">Max Other Activation: <span class="tooltip-icon">ⓘ</span></span>
                            <span class="data-value">{{{{MAX_OTHER_ACTIVATION}}}}</span>
                        </div>
                    </div>
                    
                    <div>
                        <strong data-tooltip="Bar chart of activation values for EVERY part so you can see the full activation profile of this position.">Activation in All Parts: <span class="tooltip-icon">ⓘ</span></strong>
                        {{{{PART_ACTIVATIONS}}}}
                    </div>
                </div>
                
                <div class="analysis-card">
                    <h3>📁 File References</h3>
                    <div class="data-item">
                        <span class="data-label" data-tooltip="Original Smart-Game-Format game file from which this position was extracted.">SGF File: <span class="tooltip-icon">ⓘ</span></span>
                        <span class="data-value"><a href="{{{{SGF_FILE_LINK}}}}">{{{{SGF_FILE}}}}</a></span>
                    </div>
                    <div class="data-item">
                        <span class="data-label" data-tooltip="NumPy binary file containing the encoded board tensor used as input to the model.">Board Tensor: <span class="tooltip-icon">ⓘ</span></span>
                        <span class="data-value"><a href="{{{{BOARD_NPY_LINK}}}}">{{{{BOARD_NPY}}}}</a></span>
                    </div>
                    <div class="data-item">
                        <span class="data-label" data-tooltip="Compressed KataGo self-play NPZ file that provided raw tensors and move statistics for this position.">NPZ Source: <span class="tooltip-icon">ⓘ</span></span>
                        <span class="data-value"><a href="{{{{NPZ_FILE_LINK}}}}">{{{{NPZ_FILE}}}}</a></span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Initialize Besogo after page load
        document.addEventListener('DOMContentLoaded', function() {{
            besogo.autoInit();
            console.log('Besogo Go board initialized for position {{{{GLOBAL_POS}}}}');
            
            // Track current move as user navigates through the game
            const currentMoveDisplay = document.getElementById('current-move-display');
            
            // Store the initial turn number from analysis data
            const initialTurnNumber = {{{{TURN_NUMBER}}}};
            
            // Function to calculate move number from root to current node
            function calculateMoveNumber(editor) {{
                if (!editor || !editor.getCurrent || !editor.getRoot) return initialTurnNumber;
                
                const root = editor.getRoot();
                const current = editor.getCurrent();
                if (!root || !current) return initialTurnNumber;
                
                let moveNumber = 0;
                let node = current;
                
                // Count moves from current node back to root
                while (node && node !== root) {{
                    if (node.move) {{
                        moveNumber++;
                    }}
                    node = node.parent;
                }}
                
                // Return the actual move number (this is correct - Turn 0 = initial position, Turn 1 = first move, etc.)
                return moveNumber;
            }}
            
            // Function to update current move display
            function updateCurrentMove() {{
                const besogoViewer = document.querySelector('.besogo-viewer');
                if (!besogoViewer || !besogoViewer.besogoEditor) return;
                
                const editor = besogoViewer.besogoEditor;
                const moveNumber = calculateMoveNumber(editor);
                currentMoveDisplay.textContent = `Turn ${{moveNumber}}`;
            }}
            
            // Wait for Besogo to initialize and then set up the listener
            function setupBesogoListener() {{
                const besogoViewer = document.querySelector('.besogo-viewer');
                if (!besogoViewer || !besogoViewer.besogoEditor) {{
                    // Try again immediately if Besogo isn't ready yet
                    setTimeout(setupBesogoListener, 50);
                    return;
                }}
                
                const editor = besogoViewer.besogoEditor;
                
                // Add listener for navigation changes
                editor.addListener(function(msg) {{
                    if (msg.navChange) {{
                        updateCurrentMove();
                    }}
                }});
                
                // Initial update - use the correct turn number from analysis data
                currentMoveDisplay.textContent = `Turn ${{initialTurnNumber}}`;
            }}
            
            // Start setup immediately without delay
            setupBesogoListener();
            
            // Keyboard navigation for position navigation
            document.addEventListener('keydown', function(event) {{
                // Only handle arrow keys if not typing in an input field
                if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {{
                    return;
                }}
                
                switch(event.key) {{
                    case 'ArrowLeft':
                        // Navigate to previous position
                        const prevLink = document.querySelector('a[href="{{{{PREV_POS_HTML}}}}"]');
                        if (prevLink && !prevLink.hasAttribute('style')) {{
                            event.preventDefault();
                            window.location.href = prevLink.href;
                        }}
                        break;
                    case 'ArrowRight':
                        // Navigate to next position
                        const nextLink = document.querySelector('a[href="{{{{NEXT_POS_HTML}}}}"]');
                        if (nextLink && !nextLink.hasAttribute('style')) {{
                            event.preventDefault();
                            window.location.href = nextLink.href;
                        }}
                        break;
                }}
            }});
        }});
    </script>
</body>
</html>'''


def generate_html_file(output_path: str, data: Dict[str, Any], board_size: int = 13) -> None:
    """Generate HTML file with the given data."""
    template = get_html_template(board_size)
    
    # Replace template variables
    html_content = template
    for key, value in data.items():
        placeholder = f"{{{{{key}}}}}"
        html_content = html_content.replace(placeholder, str(value))
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def process_position(summary_row: Dict[str, str], output_dir: str, all_positions: List[Dict[str, str]], board_size: int = 13) -> None:
    """Process a single position and generate its HTML file."""
    # Handle both human games format and selfplay format
    if 'global_pos' in summary_row:
        # Selfplay format
        global_pos = summary_row['global_pos']
        part = summary_row['part']
        rank = summary_row['rank']
    else:
        # Human games format - use part index as position identifier
        global_pos = summary_row.get('position_idx', '0')
        if global_pos == 'N/A':
            global_pos = summary_row.get('part_idx', '0')  # Use part_idx as position identifier
        part = summary_row.get('part_idx', '0')
        rank = '1'  # Human games don't have rank, use 1 as default
    
    # Find current position index and navigation data
    current_index = -1
    for i, pos in enumerate(all_positions):
        pos_global = pos.get('global_pos', pos.get('position_idx', '0'))
        pos_part = pos.get('part', pos.get('part_idx', '0'))
        pos_rank = pos.get('rank', '1')
        if pos_global == global_pos and pos_part == part and pos_rank == rank:
            current_index = i
            break
    
    # Get previous and next positions
    prev_pos = all_positions[current_index - 1] if current_index > 0 else None
    next_pos = all_positions[current_index + 1] if current_index < len(all_positions) - 1 else None
    
    # Get first and last positions
    first_pos = all_positions[0] if all_positions else None
    last_pos = all_positions[-1] if all_positions else None
    
    # Load analysis data from JSON file
    analysis_file = Path(output_dir).parent / "inspect_parts" / "part_analyses.json"
    analysis_data = {}
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            analyses = json.load(f)
        
        # Find the analysis for this position
        position_idx = int(global_pos)
        for analysis in analyses:
            if analysis.get('position_idx') == position_idx:
                analysis_data = analysis
                break
    
    # Load data files
    sgf_content = ""
    if 'sgf_file' in summary_row:
        sgf_content = load_sgf_content(summary_row['sgf_file'])
    elif analysis_data:
        # Use SGF content from analysis data
        sgf_content = analysis_data.get('sgf_content', "")
    else:
        # Human games format - create placeholder SGF content
        sgf_content = "(;FF[4]GM[1]SZ[13]AB[dd][dj][pd][pj]AW[cd][cj][nd][nj]C[Human game position])"
    
    # Load analysis data - use part-specific analysis file if it exists
    if 'analysis_file' in summary_row:
        analysis_file = summary_row['analysis_file']
        analysis_data = load_analysis_data(analysis_file)
    
    # For human games, create analysis data from JSON
    if not analysis_data and Path(output_dir).parent.joinpath("inspect_parts/part_analyses.json").exists():
        with open(Path(output_dir).parent / "inspect_parts/part_analyses.json", 'r') as f:
            analyses = json.load(f)
        
        # Find the analysis for this position
        position_idx = int(global_pos)
        for analysis in analyses:
            if analysis.get('position_idx') == position_idx:
                # Use the actual analysis data directly
                analysis_data = analysis
                break
    
    # If still no analysis data, create minimal data
    if not analysis_data:
        analysis_data = {
            'position_idx': int(global_pos),
            'part_idx': int(part),
            'activation_strength': float(summary_row.get('activation_strength', 0)),
            'activation_percentile': 0.0,
            'move_coord': 'Unknown',
            'turn_number': 0,
            'sgf_content': "(;FF[4]GM[1]SZ[13]C[Human game position])"
        }
    
    # Extract data for template - use actual values from analysis_data
    position_info = {
        'part': int(analysis_data.get('part_idx', part)),
        'rank': int(rank),
        'turn_number': analysis_data.get('turn_number', 0),
        'move_coordinate': analysis_data.get('move_coord', 'Unknown')
    }
    
    nmf_analysis = {
        'activation_strength': float(analysis_data.get('activation_strength', 0)),
        'total_board_activity': len(analysis_data.get('channel_activity', [])),
        'channel_activity': analysis_data.get('channel_activity', [])
    }
    
    # Enhanced Go Pattern Analysis from policy data
    policy_analysis = analysis_data.get('policy_analysis', {})
    go_pattern = {
        'move_type': 'normal',  # Default for human games
        'game_phase': 'opening'  # Default for human games
        # Removed policy entropy, confidence, and top moves for human games
    }
    
    # Enhanced Part Comparison
    part_comparison = analysis_data.get('part_comparison', {})
    component_comp = {
        'activation_percentile': float(analysis_data.get('activation_percentile', 0)),
        'uniqueness_score': float(analysis_data.get('uniqueness_score', 0)),
        'part_rank': part_comparison.get('part_rank', 1),
        'max_other_activation': float(part_comparison.get('max_other_activation', 0)),
        'activation_in_other_parts': part_comparison.get('top_other_parts', [])
    }
    
    # Parse SGF for move information (Besogo handles the board display)
    turn_number = int(position_info.get('turn_number', 0))
    board_state, move_of_interest, all_moves = parse_sgf_moves(sgf_content, turn_number, board_size)
    
    # Calculate the correct display turn number (use the actual turn number from analysis)
    display_turn_number = turn_number
    
    # Generate file links for new structured format
    sgf_file_link = "#"
    board_npy_link = "#"
    npz_file_link = "#"
    
    if 'sgf_file' in summary_row:
        sgf_file_link = f"../output/{summary_row['sgf_file']}"
    if 'board_npy' in summary_row:
        board_npy_link = f"../output/{summary_row['board_npy']}"
    if 'npz_file' in position_info:
        npz_file_link = f"../output/{position_info.get('npz_file', 'Unknown')}"
    
    # Build template data
    template_data = {
        'TITLE': f"Position {global_pos} Analysis",
        'SUBTITLE': f"Part {part}, Rank {rank} - NMF Part Analysis",
        'TIMESTAMP': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'CURRENT_INDEX': current_index,
        'TOTAL_POSITIONS': len(all_positions),
        'PREV_TITLE': f"Position {prev_pos.get('global_pos', prev_pos.get('position_idx', 'N/A'))} (Part {prev_pos.get('part', prev_pos.get('part_idx', 'N/A'))}, Rank {prev_pos.get('rank', '1')})" if prev_pos else None,
        'NEXT_TITLE': f"Position {next_pos.get('global_pos', next_pos.get('position_idx', 'N/A'))} (Part {next_pos.get('part', next_pos.get('part_idx', 'N/A'))}, Rank {next_pos.get('rank', '1')})" if next_pos else None,
        'PREV_POS_HTML': f"pos_{prev_pos.get('global_pos', prev_pos.get('position_idx', 'N/A'))}_part{prev_pos.get('part', prev_pos.get('part_idx', 'N/A'))}_rank{prev_pos.get('rank', '1')}_analysis.html" if prev_pos else "#",
        'NEXT_POS_HTML': f"pos_{next_pos.get('global_pos', next_pos.get('position_idx', 'N/A'))}_part{next_pos.get('part', next_pos.get('part_idx', 'N/A'))}_rank{next_pos.get('rank', '1')}_analysis.html" if next_pos else "#",
        'PREV_DISABLED': 'style="opacity: 0.5; pointer-events: none;"' if not prev_pos else '',
        'NEXT_DISABLED': 'style="opacity: 0.5; pointer-events: none;"' if not next_pos else '',
        'FIRST_POS': first_pos.get('global_pos', first_pos.get('position_idx', 'N/A')) if first_pos else None,
        'LAST_POS': last_pos.get('global_pos', last_pos.get('position_idx', 'N/A')) if last_pos else None,
        'FIRST_TITLE': f"Position {first_pos.get('global_pos', first_pos.get('position_idx', 'N/A'))} (Part {first_pos.get('part', first_pos.get('part_idx', 'N/A'))}, Rank {first_pos.get('rank', '1')})" if first_pos else None,
        'LAST_TITLE': f"Position {last_pos.get('global_pos', last_pos.get('position_idx', 'N/A'))} (Part {last_pos.get('part', last_pos.get('part_idx', 'N/A'))}, Rank {last_pos.get('rank', '1')})" if last_pos else None,
        'FIRST_POS_HTML': f"pos_{first_pos.get('global_pos', first_pos.get('position_idx', 'N/A'))}_part{first_pos.get('part', first_pos.get('part_idx', 'N/A'))}_rank{first_pos.get('rank', '1')}_analysis.html" if first_pos else "#",
        'LAST_POS_HTML': f"pos_{last_pos.get('global_pos', last_pos.get('position_idx', 'N/A'))}_part{last_pos.get('part', last_pos.get('part_idx', 'N/A'))}_rank{last_pos.get('rank', '1')}_analysis.html" if last_pos else "#",
        'FIRST_DISABLED': 'style="opacity: 0.5; pointer-events: none;"' if not first_pos or current_index == 0 else '',
        'LAST_DISABLED': 'style="opacity: 0.5; pointer-events: none;"' if not last_pos or current_index == len(all_positions) - 1 else '',
        'PART': part,
        'RANK': rank,
        'GLOBAL_POS': global_pos,
        'TURN_NUMBER': str(display_turn_number),
        'MOVE_COORD': position_info.get('move_coordinate', analysis_data.get('move_coord', 'Unknown')),
        'SGF_CONTENT': sgf_content,  # Raw SGF for Besogo
        'SGF_FILE': summary_row.get('sgf_file', 'Human Game'),
        'SGF_FILE_LINK': sgf_file_link,
        'BOARD_NPY': summary_row.get('board_npy', 'N/A'),
        'BOARD_NPY_LINK': board_npy_link,
        'NPZ_FILE': position_info.get('npz_file', 'Human Game Data'),
        'NPZ_FILE_LINK': npz_file_link,
        
        # NMF Analysis
        'ACTIVATION_STRENGTH': format_activation_strength(nmf_analysis.get('activation_strength', 0)),
        'ACTIVATION_PERCENT': format_percentage(nmf_analysis.get('activation_strength', 0)),
        'ACTIVATION_PERCENTILE': f"{component_comp.get('activation_percentile', 0):.2f}",
        'TOTAL_BOARD_ACTIVITY': nmf_analysis.get('total_board_activity', 0),
        'CHANNEL_BARS': generate_channel_bars(nmf_analysis.get('channel_activity', [])),
        
        # Go Pattern Analysis
        'MOVE_TYPE': go_pattern.get('move_type', 'Unknown').title(),
        'GAME_PHASE': go_pattern.get('game_phase', 'Unknown').replace('_', ' ').title(),
        
        # Part Comparison
        'UNIQUENESS_SCORE': f"{component_comp.get('uniqueness_score', 0):.4f}",
        'PART_RANK': component_comp.get('part_rank', 'Unknown'),
        'MAX_OTHER_ACTIVATION': f"{component_comp.get('max_other_activation', 0):.4f}",
        'PART_ACTIVATIONS': generate_part_activations(
            component_comp.get('activation_in_other_parts', []), all_positions
        )
    }
    
    # Generate output filename - make it unique by including part and rank
    output_filename = f"pos_{global_pos}_part{part}_rank{rank}_analysis.html"
    output_path = os.path.join(output_dir, output_filename)
    
    # Generate HTML file
    generate_html_file(output_path, template_data, board_size)

def generate_index_page(summary_data: List[Dict[str, str]], output_dir: str) -> None:
    """Generate index page with all positions."""
    index_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NMF Parts Analysis - Index</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .part-section {{ margin-bottom: 30px; }}
        .part-header {{ background: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .positions-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin-top: 15px; }}
        .position-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .position-title {{ font-weight: bold; margin-bottom: 10px; }}
        .position-details {{ margin-bottom: 15px; }}
        .detail-item {{ margin: 5px 0; }}
        .detail-label {{ font-weight: bold; }}
        .view-button {{ display: inline-block; background: #007bff; color: white; padding: 8px 15px; text-decoration: none; border-radius: 3px; }}
        .view-button:hover {{ background: #0056b3; }}
    </style>
</head>
<body>
    <h1>NMF Parts Analysis - Index</h1>
    <p>Total positions: {len(summary_data)}</p>
    <div class="content">
'''
    
    # Group positions by part
    parts = {}
    for row in summary_data:
        # Handle both formats
        if 'part' in row:
            part = row['part']
        else:
            part = row.get('part_idx', '0')
        
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
        positions.sort(key=lambda x: int(x.get('rank', '1')))
        
        for pos in positions:
            # Handle both formats
            global_pos = pos.get('global_pos', pos.get('position_idx', '0'))
            if global_pos == 'N/A':
                global_pos = pos.get('part_idx', '0')
            
            part = pos.get('part', pos.get('part_idx', '0'))
            rank = pos.get('rank', '1')
            
            index_html += f'''
                <div class="position-card">
                    <div class="position-title">Position {global_pos}</div>
                    <div class="position-details">
                        <div class="detail-item">
                            <span class="detail-label">Rank:</span>
                            <span class="detail-value">{rank}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Part:</span>
                            <span class="detail-value">{part}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Activation:</span>
                            <span class="detail-value">{pos.get('activation_strength', 'N/A')}</span>
                        </div>
                    </div>
                    <a href="pos_{global_pos}_part{part}_rank{rank}_analysis.html" class="view-button">
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-file", required=True, help="CSV summary file")
    parser.add_argument("--output-dir", required=True, help="Output directory for HTML reports")
    parser.add_argument("--board-size", type=int, default=13, help="Board size (default: 13 for human games)")
    
    args = parser.parse_args()
    
    # Set up paths
    summary_file = args.summary_file
    output_dir = args.output_dir
    board_size = args.board_size
    
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
    for row in summary_data:
        process_position(row, output_dir, summary_data, board_size)
    
    # Generate index page
    generate_index_page(summary_data, output_dir)
    
    print(f"\nAll HTML reports generated in: {output_dir}")
    print("Open index.html to view all analyses")

if __name__ == "__main__":
    main() 