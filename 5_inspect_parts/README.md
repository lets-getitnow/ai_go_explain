# Step 5: Inspect NMF Parts

This step analyzes the NMF components from step 4 to identify the strongest activations and generate detailed analysis reports.

## File Organization Requirements

### Output Directory Structure
All position files must be organized in a structured output directory with the following hierarchy:

```
output/
├── pos_4683/
│   ├── analysis.json
│   ├── board.npy
│   └── game.sgf
├── pos_89/
│   ├── analysis.json
│   ├── board.npy
│   └── game.sgf
├── pos_4109/
│   ├── analysis.json
│   ├── board.npy
│   └── game.sgf
└── ... (one directory per position)
```

### File Naming Convention
- Each position gets its own directory named `pos_{global_pos}/`
- Files within each directory use consistent names:
  - `analysis.json` - Analysis data for the position
  - `board.npy` - Board tensor data
  - `game.sgf` - SGF game file

### Script Requirements
- `inspect_parts.py` must create this directory structure when generating position files
- `generate_html_reports.py` must reference files using the new structure
- All file paths in HTML reports must be relative to the output directory structure
- No files should be left in the root directory - everything goes in position-specific subdirectories

## Process

1. **Load NMF data**: Load the NMF components and activations from step 4
2. **Find strongest activations**: For each component, find the positions with the highest activation values
3. **Generate analysis**: For each position of interest, create detailed analysis including:
   - NMF component analysis
   - Go pattern analysis  
   - Component comparison data
4. **Create structured output**: Organize all files into position-specific directories
5. **Generate HTML reports**: Create interactive HTML visualizations for each position

## Input Files

- `../4_nmf_parts/nmf_components.npy` - NMF component weights
- `../4_nmf_parts/nmf_activations.npy` - NMF activation values
- `../4_nmf_parts/nmf_meta.json` - Metadata about the NMF analysis
- `../3_extract_activations/activations/pooled_meta.json` - Position metadata
- `../3_extract_activations/activations/pos_index_to_npz.txt` - NPZ file mappings

## Output Files

### Structured Output Directory
- `output/` - Main output directory containing all position data
- `output/pos_{global_pos}/` - One subdirectory per position
- `output/pos_{global_pos}/analysis.json` - Detailed analysis data
- `output/pos_{global_pos}/board.npy` - Board tensor for the position
- `output/pos_{global_pos}/game.sgf` - SGF game file

### HTML Reports
- `html_reports/` - Directory containing all HTML visualizations
- `html_reports/index.html` - Index page linking to all analyses
- `html_reports/pos_{global_pos}_analysis.html` - Individual position analysis pages

### Summary Data
- `strong_positions_summary.csv` - CSV summary of all analyzed positions

## Analysis Components

Each position analysis includes:

### NMF Component Analysis
- **Part**: Which NMF part (group of components) this activation belongs to
- **Rank**: Rank within the part by activation strength
- **Global Position**: Unique identifier for cross-referencing
- **Activation Strength**: Raw activation value (0-1)
- **Activation Percentile**: Percentile compared to all positions
- **Channel Activity**: Which convolutional channels fired above threshold

### Go Pattern Analysis  
- **Move Type**: Normal play, pass, or resign
- **Game Phase**: Opening, middle-game, or endgame
- **Policy Entropy**: Shannon entropy of move probability distribution
- **Policy Confidence**: Probability assigned to the selected move
- **Top Policy Moves**: List of best moves with probabilities

### Component Comparison
- **Uniqueness Score**: How distinct this component's activation pattern is
- **Component Rank**: Ordering by average activation strength
- **Max Other Activation**: Highest activation among other components
- **Activation in All Components**: Full activation profile across all components

## Usage

```bash
# Run the analysis
python inspect_parts.py

# Generate HTML reports
python generate_html_reports.py
```

The scripts will automatically create the required directory structure and organize all files accordingly. 