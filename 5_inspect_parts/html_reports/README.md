# Go Position Analysis - HTML Reports

This directory contains interactive HTML visualizations for all positions of interest identified in Step 5 of the NMF analysis.

## 📁 Files Generated

- **`index.html`** - Main dashboard showing all 9 positions organized by NMF part
- **`pos_{global_pos}_analysis.html`** - Individual analysis for each position (9 files total)

## 🎯 Key Features

### **Automatic Board Navigation**
Each position HTML file **automatically navigates** to the move of interest when the page loads. No manual navigation required - the Go board will display the exact position where the NMF component showed strong activation.

### **Interactive Go Board**
- **Navigation Controls**: First/Previous/Next/Last move buttons
- **Game Replay**: Step through the entire game move by move  
- **Jump to Move of Interest**: Direct link to the NMF activation position
- **Move Counter**: Shows current move number and total moves
- **7x7 Board Display**: Proper coordinate mapping from 9x9 SGF to 7x7 analysis
- **Custom SVG Rendering**: High-quality black and white stones with wooden board

### **Comprehensive Analysis Data**
Each position page displays:

#### 🧠 NMF Component Analysis
- Activation strength and percentile ranking
- Channel activity visualization
- Component comparison data

#### 🎯 Go Pattern Analysis  
- Move type and game phase classification
- Policy entropy and confidence metrics
- Top policy moves from neural network

#### 📊 Component Comparison
- Uniqueness scores
- Cross-component activation levels
- Position ranking within component

## 🚀 How to Use

1. **Start with the Index Page**:
   ```
   Open index.html in your web browser
   ```

2. **Browse by NMF Part**:
   - Part 0: Positions 1388, 2442, 1414
   - Part 1: Positions 1256, 1254, 2173  
   - Part 2: Positions 834, 766, 764

3. **Click "View Analysis →"** on any position card to see detailed analysis

4. **Explore the Go Board**:
   - The board automatically shows the move of interest
   - Use navigation controls to see game context
   - Hover over data elements for additional details

## 📋 Position Summary

| Position | Part | Rank | Move | Turn | Type |
|----------|------|------|------|------|------|
| 1388 | 0 | 1 | (3,1) | 27 | Corner |
| 2442 | 0 | 2 | (3,2) | 0 | Corner |
| 1414 | 0 | 3 | (2,3) | 53 | Corner |
| 1256 | 1 | 1 | (0,0) | 40 | Corner |
| 1254 | 1 | 2 | (0,0) | 38 | Corner |
| 2173 | 1 | 3 | (2,5) | 30 | Corner |
| 834 | 2 | 1 | PASS | 19 | Pass |
| 766 | 2 | 2 | PASS | 24 | Pass |
| 764 | 2 | 3 | (6,0) | 22 | Corner |

## 🔧 Technical Details

- **Rendering**: Custom SVG-based Go board with JavaScript navigation
- **Board Size**: 7x7 (center region extracted from 9x9 SGF files)
- **SGF Format**: Standard SGF with KataGo extensions  
- **Navigation**: Client-side JavaScript for move replay
- **Browser Compatibility**: Modern browsers (Chrome, Firefox, Safari, Edge)
- **No Server Required**: Open files directly in browser

## 📖 Analysis Interpretation

### **High Activation Strength** (>1.0)
Indicates the NMF component strongly recognizes patterns in this position

### **Channel Activity**
Shows which neural network channels are active for each component

### **Policy Moves**
Reveals what moves the neural network considers important at this position

### **Component Uniqueness**
Higher uniqueness scores suggest the component captures distinct Go patterns

## ⚠️ Critical Feature

**AUTOMATIC NAVIGATION**: Each HTML file uses the `path` parameter to automatically jump to the specific turn of interest. This ensures you immediately see the relevant board position without manual navigation.

## 🔄 Regeneration

To regenerate these files with updated data:
```bash
cd 5_inspect_parts
python3 generate_html_reports.py
```

The generator script will recreate all HTML files from the current analysis data. 