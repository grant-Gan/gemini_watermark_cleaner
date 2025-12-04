import cv2
import numpy as np
import torch
from PIL import Image
from iopaint.model import LaMa
from iopaint.schema import InpaintRequest
import os

class WatermarkRemover:
    def __init__(self, device='cpu'):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        print(f"Initializing LaMa model on {self.device}...")
        try:
            self.model = LaMa(device=self.device)
        except Exception as e:
            print(f"Error initializing model: {e}")
            self.model = None

    def detect_watermark(self, image_cv2, canny_threshold=100, dilation_width=3.0, roi_ratio=(0.3, 0.15)):
        """
        Automatically detects watermark in corners.
        canny_threshold: Threshold for edge detection (sensitivity).
        dilation_width: Width of the horizontal dilation kernel (expansion).
        roi_ratio: Tuple (width_pct, height_percent) defining the search box anchored at Bottom-Right.
        """
        h, w = image_cv2.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # ROI Dimensions relative to bottom-right
        r_w, r_h = roi_ratio
        # Fallback if 0
        if r_w <= 0: r_w = 0.3
        if r_h <= 0: r_h = 0.15
            
        w_margin = int(w * r_w)
        h_margin = int(h * r_h)
        
        # Sanity check
        w_margin = max(10, min(w, w_margin))
        h_margin = max(10, min(h, h_margin))
        
        # Bottom Right ROI
        # Coords: y from h-h_margin to h, x from w-w_margin to w
        br_roi = image_cv2[h-h_margin:h, w-w_margin:w]
        
        def get_edges(roi, thresh):
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, thresh, thresh * 2.5)
            return edges

        br_edges = get_edges(br_roi, canny_threshold)
        
        # Dilation settings
        k_w = max(1, int(round(dilation_width)))
        kernel_h = np.ones((1, k_w), np.uint8) 
        
        def process_roi(roi_edges, roi_x_offset, roi_y_offset):
            if dilation_width >= 1.0:
                dilated = cv2.dilate(roi_edges, kernel_h, iterations=1)
            else:
                dilated = roi_edges
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            found_any = False
            
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                if bw < 5 or bh < 5: continue
                # Relative to ROI width, not hardcoded 30% anymore
                if bw > (w_margin * 0.9): continue 
                if bh > bw * 2: continue

                found_any = True
                pad = 2
                
                g_x1 = roi_x_offset + x - pad
                g_y1 = roi_y_offset + y - pad
                g_x2 = roi_x_offset + x + bw + pad
                g_y2 = roi_y_offset + y + bh + pad
                
                g_x1 = max(0, g_x1)
                g_y1 = max(0, g_y1)
                g_x2 = min(w, g_x2)
                g_y2 = min(h, g_y2)
                
                cv2.rectangle(mask, (g_x1, g_y1), (g_x2, g_y2), 255, -1)
            
            return found_any

        found_mask = False
        min_pixel_trigger = 10 
        
        br_score = np.count_nonzero(br_edges)
        
        if br_score > min_pixel_trigger:
            # Offset is Top-Left of ROI in Global Image
            if process_roi(br_edges, w - w_margin, h - h_margin):
                found_mask = True
        
        # Note: Bottom-Left detection is disabled as user requested "Only process inside red box" 
        # and the red box is explicitly "Bottom-Right anchored".
        
        # Fallback
        if not found_mask:
            print("No clear watermark detected. Applying default small mask.")
            box_w = min(200, w_margin)
            box_h = min(50, h_margin)
            cv2.rectangle(mask, (w-box_w, h-box_h), (w, h), 255, -1)
            
        return mask

    def process_image(self, input_path, output_path, threshold=100, dilation_iter=3.0, roi_ratio=(0.3, 0.15)):
        if not self.model:
            print("Model not loaded.")
            return False

        img = cv2.imread(input_path)
        if img is None:
            print(f"Could not load image: {input_path}")
            return False
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = self.detect_watermark(
            img, 
            canny_threshold=threshold, 
            dilation_width=dilation_iter,
            roi_ratio=roi_ratio
        )
        
        config = InpaintRequest()
        
        try:
            res_bgr = self.model(img_rgb, mask, config)
            cv2.imwrite(output_path, res_bgr)
            print(f"Processed: {input_path} -> {output_path}")
            return True
        except Exception as e:
            print(f"Inpainting failed: {e}")
            return False

if __name__ == "__main__":
    # Test
    remover = WatermarkRemover()
    # remover.process_image("test1.png", "test1_cleaned.png")
