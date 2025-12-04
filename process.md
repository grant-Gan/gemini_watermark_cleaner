# Process Log - Gemini Watermark Remover

## Initialization
- Date: 2025-12-04
- Goal: Create a GUI application to remove Gemini watermarks using IOPaint.

## Steps Taken
1.  **Initialization**: Created this log file.
2.  **Analysis**: Checked project structure. Found `test1.png` and `test2.png` in root.
3.  **Environment Setup**: 
    - Executed `uv add iopaint PyQt6 opencv-python pillow torch torchvision`.
    - Dependencies installed successfully.
4.  **Code Implementation**:
    - **`watermark_remover.py`**: Implemented `WatermarkRemover` class using `iopaint`'s `LaMa` model. Added `detect_watermark` function.
    - **`gui.py`**: Created a PyQt6-based GUI.
    - **`main.py`**: Entry point.
    - **`auto_test.py`**: Script to run function tests.
5.  **Testing & Debugging**:
    - **Issue 1**: Colors inverted.
        - **Fix**: Verified LaMa returns BGR, updated code to save directly.
    - **Issue 2**: "Large smear" artifacts.
        - **Fix**: Increased threshold (100), minimized dilation (1x2), filtered long lines.
    - **Verification**: `verify_mask_size.py` showed mask size reduced to < 4% with good removal.
    - **Issue 3**: `NameError: name 'sys' is not defined`.
        - **Fix**: Added `import sys` to `gui.py`.
6.  **Feature Enhancement**:
    - **GUI Update**:
        - Added "Detection Parameters" group.
        - Added "Input Folder" & "Output Folder" selection.
        - Added `BatchWorker` thread.
        - Added `QProgressBar`.
    - **Sidebar Enhancement**:
        - Split layout into Sidebar (`QListWidget`) and Main Content.
        - Added "Files to Process" list with Add/Remove buttons.
        - Implemented batch processing on the list items.
        - Implemented preview sync on list item click.
    - **Zoom & Preview Enhancement**:
        - **Custom Widget**: Created `ImagePreviewWidget`.
        - **Features**: Zoom (Wheel), Pan (Drag), Fit Button, 'Z' key shortcut.
    - **Parameter Logic Refactor**:
        - GUI inputs converted to Float.
        - Logic updated to use Float Kernel Width and Float Threshold.
    - **ROI Control Enhancement**:
        - **GUI**: Added `QSlider` for Width% and Height% to define a manual Region of Interest (ROI).
        - **Visualization**: `ImagePreviewWidget` now draws a red semi-transparent box to show the ROI.
        - **Logic**: `watermark_remover.py` logic updated to accept `roi_ratio` and only detect/mask within that specific area relative to the bottom-right corner.
        - **Bug Fix**: Fixed indentation error in `gui.py` where methods were nested inside `select_output_folder`.

## Usage
- **Run GUI**: `uv run python main.py`
- **Run Tests**: `uv run python auto_test.py`
