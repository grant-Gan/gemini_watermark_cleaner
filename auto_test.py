import os
import sys
import cv2
import numpy as np
from watermark_remover import WatermarkRemover

def calculate_mse(img1, img2):
    h, w = img1.shape[:2]
    diff = cv2.absdiff(img1, img2)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th, thres = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    non_zero_count = cv2.countNonZero(thres)
    mse = np.sum(diff**2) / float(h*w)
    return mse, non_zero_count

def run_tests():
    print("Starting Auto Test with Verification...")
    remover = WatermarkRemover()
    
    test_files = ["test1.png", "test2.png"]
    
    if not os.path.exists("test_results"):
        os.makedirs("test_results")
        
    all_passed = True
    
    for f in test_files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            continue
            
        output_path = os.path.join("test_results", f"cleaned_{f}")
        print(f"\nProcessing {f} -> {output_path}")
        
        success = remover.process_image(f, output_path)
        if not success:
            print(f"FAILED: Processing returned False for {f}")
            all_passed = False
            continue
            
        # Verification
        img_orig = cv2.imread(f)
        img_processed = cv2.imread(output_path)
        
        if img_orig.shape != img_processed.shape:
            print("FAILED: Shape mismatch.")
            all_passed = False
            continue
            
        h, w = img_orig.shape[:2]
        
        # 1. Check Main Body (should be unchanged)
        # We exclude bottom 15%
        main_h = int(h * 0.85)
        roi_main_orig = img_orig[0:main_h, :]
        roi_main_proc = img_processed[0:main_h, :]
        
        mse_main, diff_pixels_main = calculate_mse(roi_main_orig, roi_main_proc)
        print(f"  Main Body MSE: {mse_main:.4f} (Diff Pixels: {diff_pixels_main})")
        
        if mse_main > 5.0: # Allow slight compression noise
            print("  FAILED: Main body of image changed significantly! Color space issue?")
            all_passed = False
        else:
            print("  PASS: Main body preserved.")
            
        # 2. Check Watermark Area (Bottom Right)
        # We expect SOME change here.
        box_h = 80
        box_w = 300
        roi_wm_orig = img_orig[h-box_h:h, w-box_w:w]
        roi_wm_proc = img_processed[h-box_h:h, w-box_w:w]
        
        mse_wm, diff_pixels_wm = calculate_mse(roi_wm_orig, roi_wm_proc)
        print(f"  Watermark Area MSE: {mse_wm:.4f} (Diff Pixels: {diff_pixels_wm})")
        
        if mse_wm < 0.1:
            print("  WARNING: Watermark area barely changed. Watermark might still be there.")
            # We don't fail here because maybe the watermark was already empty?
            # But for this specific test we expect a change.
            if "test" in f: 
                 print("  FAILED: No change detected in watermark area.")
                 all_passed = False
        else:
            print("  PASS: Change detected in watermark area.")

    if all_passed:
        print("\nALL TESTS PASSED.")
    else:
        print("\nSOME TESTS FAILED.")

if __name__ == "__main__":
    run_tests()