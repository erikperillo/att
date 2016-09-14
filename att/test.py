#!/usr/bin/env python3

import im
import feat as ft
import cvx
import cv2
import os
import oarg

def error(msg, code=1):
    """
    Prints error message and exits with code.
    """
    print("error:", msg)
    exit(code)

def str_to_list(string, tp, delim=","):
    """
    Divides string separated by delimiter and converts to tp.
    """
    return list(map(tp, string.split(delim)))

def main():
    #command-line arguments
    img_file = oarg.Oarg("-i --img", "", "path to image", 0) 
    pyr_lvl = oarg.Oarg("-p --pyr-lvl", 3, "levels for pyramid") 
    #must be comma-separated
    str_cs_ksizes= oarg.Oarg("-c --cs-ks", "3,7", 
        "center-surround kernel sizes") 
    #must be comma-separated
    str_feats = oarg.Oarg("-f --features", 
        "l1,l0,r,g,b,y,hor,ver,l_diag,r_diag", "features to use")
    max_w = oarg.Oarg("-W --max-w", 800, "maximum width for image")
    max_h = oarg.Oarg("-H --max-h", 600, "maximum hwight for image")
    debug = oarg.Oarg("-d --debug", True, "debug mode")
    list_fts = oarg.Oarg("-l --list-features", False, "list available features")
    save_dir = oarg.Oarg("-s --save-dir", ".", "directory to save images")
    hlp = oarg.Oarg("-h --help", False, "this help message")

    #parsing args
    oarg.parse()

    #help message
    if hlp.val:
        oarg.describeArgs("options:", def_val=True)
        exit()

    if list_fts.val:
        print("features:", ", ".join(ft.get_available_features()))
        exit()

    #checking validity of args
    if not img_file.found:
        error("image file not found (use -h for help)")

    img = cv2.imread(img_file.val)

    #checking validity of image
    if img is None:
        error("could not read image")
    if len(img.shape) < 3:
        error("image must be colored")

    print("on file %s" % img_file.val)
    
    #resizing image
    img = cvx.resize(img, max_w.val, max_h.val)
    #adapting image dimensions for proper pyramid up/downscaling
    img = cvx.pyr_prepare(img, pyr_lvl.val)

    #displaying original image
    cvx.display(img, "original image", False)

    #getting center-surround kernel sizes
    cs_ksizes = str_to_list(str_cs_ksizes.val, int)
    feats = str_to_list(str_feats.val, str)

    #getting file name without extension
    f_name = ".".join(os.path.basename(img_file.val).split(".")[:-1])

    ims = []
    #computing intensity maps
    for feature in feats:
        print("\ton feature '%s' ..." % feature)
        input_img = ft.get_feature(img, feature)
        db_im, imap = im.intensity_map(input_img, pyr_lvl.val, cs_ksizes,    
            debug=debug.val)
        
        #displaying images
        cvx.display(input_img, "input image %s" % feature)
        cvx.display(imap, "intensity map %s" % feature)
        if debug:
            cvx.display(db_im, "intermediary intensity_map %s" % feature)

        #appending partial intensity map
        ims.append(imap)

        #saving images
        if save_dir.val:
            cvx.save(imap, "%s/%s_%s_map.png" % (save_dir.val, f_name, feature))
            if debug:
                cvx.save(db_im, "%s/%s_%s_db_map.png" % \
                    (save_dir.val, f_name, feature))

    #computing final intensity map
    final_im = sum(ims)
    
    #displaying final result
    cvx.display(final_im, "final intensity map")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #saving final result
    if save_dir.val:
        cvx.save(final_im, "%s/%s_final_map.png" % (save_dir.val, f_name))

if __name__ == "__main__":
    main()
