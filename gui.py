import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox,
                             QProgressBar, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit,
                             QListWidget, QListWidgetItem, QAbstractItemView, QSlider)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint
import cv2
from watermark_remover import WatermarkRemover

class ImagePreviewWidget(QWidget):
    def __init__(self, placeholder_text="Image"):
        super().__init__()
        self.pixmap = None
        self.scale_factor = 1.0
        self.pan_pos = QPoint(0, 0)
        self.is_panning = False
        self.last_mouse_pos = QPoint()
        self.placeholder_text = placeholder_text
        
        self.roi_ratio = (0.0, 0.0) # (width_ratio, height_ratio)
        
        # Fit Button (Floating top right)
        self.fit_btn = QPushButton("Fit", self)
        self.fit_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self.fit_btn.clicked.connect(self.fit_to_view)
        self.fit_btn.setFixedSize(50, 30)
        self.fit_btn.setStyleSheet("background-color: rgba(255, 255, 255, 0.8); border: 1px solid #999;")
        self.fit_btn.hide() 
        
        # Style
        self.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_roi_ratio(self, w, h):
        self.roi_ratio = (w, h)
        self.update()

    def enterEvent(self, event):
        self.setFocus()
        super().enterEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Z:
            self.fit_to_view()
        else:
            super().keyPressEvent(event)

    def set_image(self, path):
        if not path or not os.path.exists(path):
            self.pixmap = None
            self.fit_btn.hide()
            self.update()
            return

        self.pixmap = QPixmap(path)
        if not self.pixmap.isNull():
            self.fit_to_view()
            self.fit_btn.show()
            self.update()
        else:
            self.pixmap = None
            self.fit_btn.hide()
            self.update()

    def fit_to_view(self):
        if not self.pixmap:
            return
        
        # Calculate scale to fit
        w_ratio = self.width() / self.pixmap.width()
        h_ratio = self.height() / self.pixmap.height()
        self.scale_factor = min(w_ratio, h_ratio) * 0.95 
        # Clamp to reasonable initial zoom, but allow min 10% max 500% per req
        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))
        
        self.center_image()
        self.update()

    def center_image(self):
        if not self.pixmap:
            return
        img_w = self.pixmap.width() * self.scale_factor
        img_h = self.pixmap.height() * self.scale_factor
        x = (self.width() - img_w) / 2
        y = (self.height() - img_h) / 2
        self.pan_pos = QPoint(int(x), int(y))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Draw background text if no image
        if not self.pixmap or self.pixmap.isNull():
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder_text)
            return

        # Draw pixmap
        w = int(self.pixmap.width() * self.scale_factor)
        h = int(self.pixmap.height() * self.scale_factor)
        painter.drawPixmap(self.pan_pos.x(), self.pan_pos.y(), w, h, self.pixmap)
        
        # Draw ROI Box if active
        roi_w_ratio, roi_h_ratio = self.roi_ratio
        if roi_w_ratio > 0 and roi_h_ratio > 0:
            # Original image dims
            orig_w = self.pixmap.width()
            orig_h = self.pixmap.height()
            
            # ROI in pixels relative to image
            box_w_px = int(orig_w * roi_w_ratio)
            box_h_px = int(orig_h * roi_h_ratio)
            
            # ROI Top-Left in Image Space (Anchored Bottom-Right)
            # x = width - box_w
            # y = height - box_h
            img_x = orig_w - box_w_px
            img_y = orig_h - box_h_px
            
            # Transform to View Space
            # view_x = pan.x + img_x * scale
            view_x = self.pan_pos.x() + int(img_x * self.scale_factor)
            view_y = self.pan_pos.y() + int(img_y * self.scale_factor)
            view_w = int(box_w_px * self.scale_factor)
            view_h = int(box_h_px * self.scale_factor)
            
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(view_x, view_y, view_w, view_h)
            
            # Draw fill with alpha
            painter.fillRect(view_x, view_y, view_w, view_h, QColor(255, 0, 0, 50))

    def wheelEvent(self, event):
        if not self.pixmap:
            return
            
        # Zoom logic
        zoom_in = event.angleDelta().y() > 0
        multiplier = 1.1 if zoom_in else 0.9
        
        new_scale = self.scale_factor * multiplier
        # Limit 10% to 500%
        new_scale = max(0.1, min(new_scale, 5.0)) 
        
        # Zoom relative to center of view to keep image stable-ish
        view_center = QPoint(self.width() // 2, self.height() // 2)
        
        # Image coordinate under the center of the view
        img_x = (view_center.x() - self.pan_pos.x()) / self.scale_factor
        img_y = (view_center.y() - self.pan_pos.y()) / self.scale_factor
        
        self.scale_factor = new_scale
        
        # Calculate new pan position to keep that image coordinate at center
        new_pan_x = view_center.x() - img_x * self.scale_factor
        new_pan_y = view_center.y() - img_y * self.scale_factor
        
        self.pan_pos = QPoint(int(new_pan_x), int(new_pan_y))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.pos() - self.last_mouse_pos
            self.pan_pos += delta
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False

    def resizeEvent(self, event):
        # Move fit button to top right
        btn_w = self.fit_btn.width()
        self.fit_btn.move(self.width() - btn_w - 10, 10)
        super().resizeEvent(event)

class Worker(QThread):
    finished = pyqtSignal(bool, str)
    
    def __init__(self, remover, input_path, output_path, threshold, dilation, roi_ratio):
        super().__init__()
        self.remover = remover
        self.input_path = input_path
        self.output_path = output_path
        self.threshold = threshold
        self.dilation = dilation
        self.roi_ratio = roi_ratio
        
    def run(self):
        try:
            success = self.remover.process_image(
                self.input_path, 
                self.output_path, 
                threshold=self.threshold, 
                dilation_iter=self.dilation,
                roi_ratio=self.roi_ratio
            )
            if success:
                self.finished.emit(True, self.output_path)
            else:
                self.finished.emit(False, "Processing failed.")
        except Exception as e:
            self.finished.emit(False, str(e))

class BatchWorker(QThread):
    image_started = pyqtSignal(str)
    image_finished = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    batch_finished = pyqtSignal(bool, str)
    
    def __init__(self, remover, input_files, output_dir, threshold, dilation, roi_ratio):
        super().__init__()
        self.remover = remover
        self.input_files = input_files
        self.output_dir = output_dir
        self.threshold = threshold
        self.dilation = dilation
        self.roi_ratio = roi_ratio
        self.is_running = True

    def run(self):
        count = 0
        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
            except Exception as e:
                self.batch_finished.emit(False, f"Could not create output directory: {e}")
                return

        for fpath in self.input_files:
            if not self.is_running:
                break
            
            self.image_started.emit(fpath)
            
            fname = os.path.basename(fpath)
            output_path = os.path.join(self.output_dir, fname)
            
            try:
                success = self.remover.process_image(
                    fpath, 
                    output_path, 
                    threshold=self.threshold, 
                    dilation_iter=self.dilation,
                    roi_ratio=self.roi_ratio
                )
                if success:
                    self.image_finished.emit(output_path)
                else:
                    pass
            except Exception as e:
                print(f"Error processing {fname}: {e}")
            
            count += 1
            self.progress_updated.emit(count)
            
        self.batch_finished.emit(True, "Batch processing complete.")

    def stop(self):
        self.is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Watermark Cleaner")
        self.resize(1400, 800)
        
        self.remover = None
        self.current_image_path = None
        self.processed_image_path = "processed_temp.png"
        self.output_folder_path = None
        
        self.init_ui()
        
        self.status_label.setText("Initializing AI Model (this may take time)...")
        
        self.init_thread = InitThread()
        self.init_thread.finished.connect(self.on_model_loaded)
        self.init_thread.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Root layout is Horizontal (Sidebar | Main Content)
        root_layout = QHBoxLayout(central_widget)
        
        # --- Sidebar ---
        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        sidebar_label = QLabel("Files to Process")
        sidebar_layout.addWidget(sidebar_label)
        
        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.file_list_widget.itemClicked.connect(self.on_file_list_clicked)
        sidebar_layout.addWidget(self.file_list_widget)
        
        sidebar_btns_layout = QHBoxLayout()
        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.clicked.connect(self.add_files_to_list)
        self.btn_remove_file = QPushButton("Remove")
        self.btn_remove_file.clicked.connect(self.remove_file_from_list)
        
        sidebar_btns_layout.addWidget(self.btn_add_files)
        sidebar_btns_layout.addWidget(self.btn_remove_file)
        sidebar_layout.addLayout(sidebar_btns_layout)
        
        root_layout.addWidget(sidebar_widget)
        
        # --- Main Content (Right Side) ---
        main_content_widget = QWidget()
        main_layout = QVBoxLayout(main_content_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Top: Settings & Folders ---
        top_layout = QHBoxLayout()
        
        # 1. Parameters Group
        params_group = QGroupBox("Detection Parameters")
        params_layout = QFormLayout()
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(1.0, 500.0)
        self.threshold_spin.setValue(100.0)
        self.threshold_spin.setSingleStep(5.0)
        self.threshold_spin.setToolTip("Canny Edge Threshold. Higher = Stricter detection (less noise).")
        
        self.dilation_spin = QDoubleSpinBox()
        self.dilation_spin.setRange(0.0, 20.0)
        self.dilation_spin.setValue(3.0)
        self.dilation_spin.setSingleStep(0.5)
        self.dilation_spin.setToolTip("Mask Expansion Width (pixels). Floating point supported.")
        
        params_layout.addRow("Edge Threshold:", self.threshold_spin)
        params_layout.addRow("Mask Expansion:", self.dilation_spin)
        params_group.setLayout(params_layout)
        
        top_layout.addWidget(params_group)
        
        # 1.5 Region of Interest Group
        roi_group = QGroupBox("Region of Interest (Bottom-Right)")
        roi_layout = QFormLayout()
        
        # Width Slider
        self.roi_w_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_w_slider.setRange(0, 100) # 0% to 100% of image width
        self.roi_w_slider.setValue(30) # Default 30%
        self.roi_w_label = QLabel("30%")
        self.roi_w_slider.valueChanged.connect(self.update_roi_preview)
        
        w_layout = QHBoxLayout()
        w_layout.addWidget(self.roi_w_slider)
        w_layout.addWidget(self.roi_w_label)
        
        # Height Slider
        self.roi_h_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_h_slider.setRange(0, 100) # 0% to 100%
        self.roi_h_slider.setValue(15) # Default 15%
        self.roi_h_label = QLabel("15%")
        self.roi_h_slider.valueChanged.connect(self.update_roi_preview)
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.roi_h_slider)
        h_layout.addWidget(self.roi_h_label)
        
        roi_layout.addRow("Width %:", w_layout)
        roi_layout.addRow("Height %:", h_layout)
        roi_group.setLayout(roi_layout)
        
        top_layout.addWidget(roi_group)
        
        # 2. Folder/File Selection Group
        files_group = QGroupBox("File & Folder Operations")
        files_layout = QVBoxLayout()
        
        # Input
        input_layout = QHBoxLayout()
        self.batch_input_btn = QPushButton("Load Folder to Sidebar")
        self.batch_input_btn.clicked.connect(self.select_input_folder)
        
        input_layout.addWidget(self.batch_input_btn)
        files_layout.addLayout(input_layout)
        
        # Output
        output_layout = QHBoxLayout()
        self.output_line = QLineEdit()
        self.output_line.setPlaceholderText("Output Folder (Default: input_folder/cleaned)")
        self.output_line.setReadOnly(True)
        
        self.output_btn = QPushButton("Select Output Folder")
        self.output_btn.clicked.connect(self.select_output_folder)
        
        output_layout.addWidget(self.output_line)
        output_layout.addWidget(self.output_btn)
        files_layout.addLayout(output_layout)
        
        files_group.setLayout(files_layout)
        top_layout.addWidget(files_group)
        
        main_layout.addLayout(top_layout)
        
        # --- Middle: Image Preview (Zoomable) ---
        image_layout = QHBoxLayout()
        
        # Replaced QLabel with ImagePreviewWidget
        self.original_widget = ImagePreviewWidget("Original Image")
        self.result_widget = ImagePreviewWidget("Result Image")
        
        image_layout.addWidget(self.original_widget)
        image_layout.addWidget(self.result_widget)
        
        main_layout.addLayout(image_layout)
        
        # --- Bottom: Actions & Status ---
        bottom_layout = QVBoxLayout()
        
        action_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("Process Selected Image")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumHeight(40)
        
        self.batch_process_btn = QPushButton("Batch Process All in List")
        self.batch_process_btn.clicked.connect(self.process_batch)
        self.batch_process_btn.setEnabled(False)
        self.batch_process_btn.setMinimumHeight(40)
        
        action_layout.addWidget(self.process_btn)
        action_layout.addWidget(self.batch_process_btn)
        bottom_layout.addLayout(action_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        bottom_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Starting...")
        bottom_layout.addWidget(self.status_label)
        
        main_layout.addLayout(bottom_layout)
        
        root_layout.addWidget(main_content_widget)

        # Disable inputs until model loads
        self.btn_add_files.setEnabled(False)
        self.btn_remove_file.setEnabled(False)
        self.batch_input_btn.setEnabled(False)
        self.output_btn.setEnabled(False)

    def on_model_loaded(self, remover):
        if remover:
            self.remover = remover
            self.status_label.setText("Model Ready. Load an image or folder.")
            self.btn_add_files.setEnabled(True)
            self.btn_remove_file.setEnabled(True)
            self.batch_input_btn.setEnabled(True)
            self.output_btn.setEnabled(True)
        else:
            self.status_label.setText("Model Initialization Failed.")
            QMessageBox.critical(self, "Error", "Failed to initialize AI model.")

    def add_files_to_list(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, "Add Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fnames:
            for f in fnames:
                existing_items = [self.file_list_widget.item(i).text() for i in range(self.file_list_widget.count())]
                if f not in existing_items:
                    self.file_list_widget.addItem(f)
            
            self.update_batch_ui_state()
            self.status_label.setText(f"Added {len(fnames)} files.")

    def remove_file_from_list(self):
        row = self.file_list_widget.currentRow()
        if row >= 0:
            self.file_list_widget.takeItem(row)
            self.current_image_path = None
            self.process_btn.setEnabled(False)
            self.update_batch_ui_state()

    def on_file_list_clicked(self, item):
        fpath = item.text()
        if os.path.exists(fpath):
            self.current_image_path = fpath
            self.display_image(fpath, self.original_widget)
            self.result_widget.set_image(None) # Clear previous result
            self.status_label.setText(f"Selected: {os.path.basename(fpath)}")
            self.process_btn.setEnabled(True)
        else:
            self.status_label.setText("File not found.")

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.file_list_widget.clear()
            
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
            
            if not files:
                QMessageBox.warning(self, "No Images", "No supported images found in this folder.")
                return
            
            self.file_list_widget.addItems(files)
            self.update_batch_ui_state()
            self.status_label.setText(f"Loaded folder: {len(files)} images.")
            
            if not self.output_folder_path:
                self.output_folder_path = os.path.join(folder, "cleaned")
                self.output_line.setText(self.output_folder_path)

    def update_batch_ui_state(self):
        has_files = self.file_list_widget.count() > 0
        self.batch_process_btn.setEnabled(has_files)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_path = folder
            self.output_line.setText(folder)

    def update_roi_preview(self):
        w_percent = self.roi_w_slider.value()
        h_percent = self.roi_h_slider.value()
        self.roi_w_label.setText(f"{w_percent}%")
        self.roi_h_label.setText(f"{h_percent}%")
        
        # Update preview widget (ratio 0.0-1.0)
        self.original_widget.set_roi_ratio(w_percent / 100.0, h_percent / 100.0)

    def process_image(self):
        if not self.current_image_path or not self.remover:
            return
        
        self.status_label.setText("Processing... Please wait.")
        self.set_controls_enabled(False)
        
        roi = (self.roi_w_slider.value() / 100.0, self.roi_h_slider.value() / 100.0)
        
        self.worker = Worker(
            self.remover, 
            self.current_image_path, 
            self.processed_image_path,
            self.threshold_spin.value(),
            self.dilation_spin.value(),
            roi
        )
        self.worker.finished.connect(self.on_process_finished)
        self.worker.start()

    def on_process_finished(self, success, message):
        self.set_controls_enabled(True)
        
        if success:
            self.status_label.setText("Processing Complete.")
            self.display_image(message, self.result_widget)
        else:
            self.status_label.setText(f"Error: {message}")
            QMessageBox.warning(self, "Processing Error", message)

    def process_batch(self):
        count = self.file_list_widget.count()
        if count == 0 or not self.remover:
            return
            
        files_to_process = [self.file_list_widget.item(i).text() for i in range(count)]
            
        if not self.output_folder_path:
            # Try to determine default from first file
            first_dir = os.path.dirname(files_to_process[0])
            self.output_folder_path = os.path.join(first_dir, "cleaned")
            self.output_line.setText(self.output_folder_path)
            
        self.status_label.setText("Batch Processing Started...")
        self.set_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(count)
        self.progress_bar.setValue(0)
        
        roi = (self.roi_w_slider.value() / 100.0, self.roi_h_slider.value() / 100.0)
        
        self.batch_worker = BatchWorker(
            self.remover,
            files_to_process,
            self.output_folder_path,
            self.threshold_spin.value(),
            self.dilation_spin.value(),
            roi
        )
        self.batch_worker.image_started.connect(self.on_batch_image_started)
        self.batch_worker.image_finished.connect(self.on_batch_image_finished)
        self.batch_worker.progress_updated.connect(self.progress_bar.setValue)
        self.batch_worker.batch_finished.connect(self.on_batch_finished)
        self.batch_worker.start()

    def on_batch_image_started(self, path):
        self.display_image(path, self.original_widget)
        self.status_label.setText(f"Processing: {os.path.basename(path)}")
        items = self.file_list_widget.findItems(path, Qt.MatchFlag.MatchExactly)
        if items:
            self.file_list_widget.setCurrentItem(items[0])

    def on_batch_image_finished(self, path):
        self.display_image(path, self.result_widget)

    def on_batch_finished(self, success, message):
        self.set_controls_enabled(True)
        self.status_label.setText(message)
        QMessageBox.information(self, "Batch Complete", message)
        self.progress_bar.setVisible(False)

    def set_controls_enabled(self, enabled):
        self.btn_add_files.setEnabled(enabled)
        self.btn_remove_file.setEnabled(enabled)
        self.batch_input_btn.setEnabled(enabled)
        self.output_btn.setEnabled(enabled)
        self.process_btn.setEnabled(enabled and self.current_image_path is not None)
        self.batch_process_btn.setEnabled(enabled and self.file_list_widget.count() > 0)
        self.threshold_spin.setEnabled(enabled)
        self.dilation_spin.setEnabled(enabled)
        self.file_list_widget.setEnabled(enabled)

    def display_image(self, path, widget):
        # Updates the ImagePreviewWidget
        widget.set_image(path)

class InitThread(QThread):
    finished = pyqtSignal(object)
    
    def run(self):
        try:
            remover = WatermarkRemover()
            self.finished.emit(remover)
        except Exception:
            self.finished.emit(None)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()