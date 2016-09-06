#!/usr/bin/env python2.7

import cv2
import numpy as np
import oarg

#functions for getting components of Lab image. assumes image is already in Lab
LAB_ATTR_FUNCS = {
    "l1": lambda x: x[:, :, 0],
    "l0": lambda x: inv(LAB_ATTR_FUNCS["l1"](x)),
    "r": lambda x: x[:, :, 1],
    "g": lambda x: inv(LAB_ATTR_FUNCS["r"](x)),
    "y": lambda x: x[:, :, 2],
    "b": lambda x: inv(LAB_ATTR_FUNCS["y"](x))
}

class InvalidDimensions(Exception):
    """Exception for unexpected/invalid dimensions."""
    pass

class InvalidDataType(Exception):
    """Exception for invalid data types."""
    pass

def inv(img):
    """Inverts image."""
    return img.max() - img

def get_lab_attr(img, attr, cvt=True):
    """Gets Lab colorspace plane attr from image. Assumes img is in BGR."""
    #converting image to lab if required
    if cvt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    attr = attr.lower()
    #checking validity of argument
    if not attr in LAB_ATTR_FUNCS:
        raise TypeError("invalid attr '%s' (use: %s)" % \
            (attr, ",".join(LAB_ATTR_FUNCS)))

    return LAB_ATTR_FUNCS[attr](img)

def is_even(num):
    return num % 2 == 0

def mk_divisable(img, div):
    """Reshapes image so as to be divisible by div."""
    y_clip = img.shape[0] % div
    x_clip = img.shape[1] % div

    return img[y_clip:, x_clip:]

def pyr_prepare(img, pyr_lvl):
    """Reshapes image so as to be down/up sampled and keep original shape."""
    return mk_divisable(img, 2**pyr_lvl)

def get_center_surround_kernel(size, dtype=np.float32):
    """Gets kernel to be used in convolution to perform center-surround
        operation on image."""
    size = int(size)

    #checking validity of arguments
    if is_even(size):
        raise InvalidDimensions("kernel size must be odd")
    if size < 2:
        raise InvalidDimensions("kernel size must be bigger than 1")
    if dtype != np.float32 and dtype != np.float64:
        raise InvalidDataType("kernel dtpe must be either float32 or float64")

    #creating kernel
    weight = -1.0/(size**2 - 1)
    kernel = weight*np.ones(dtype=dtype, shape=(size, size))
    kernel[size/2, size/2] = 1.0

    return kernel

def center_surround(img, kernel):
    """Performs center-surround operation on image with given kernel."""
    #checking validity of image dimensions
    if len(img.shape) > 2:
        raise InvalidDimensions("image must have depth one")

    #applying operation through image
    ddepth = cv2.CV_32F if kernel.dtype == np.float32 else cv2.CV_64F
    cs_img = cv2.filter2D(img, ddepth, kernel)

    return cs_img.clip(min=0.0)

def rep_pyrUp(img, n):
    """Performs pyrUp repeated times."""
    for __ in xrange(n):
        img = cv2.pyrUp(img)

    return img

def fill(img, width, height, const=0):
    """Fills image with const value in order to get specified dimensions."""
    h_diff = height - img.shape[0]
    if h_diff > 0:
        filling = const*np.ones(dtype=img.dtype, shape=(h_diff, img.shape[1]))
        img = np.vstack((img, filling))

    w_diff = width - img.shape[1]
    if w_diff > 0:
        filling = const*np.ones(dtype=img.dtype, shape=(img.shape[0], w_diff))
        img = np.hstack((img, filling))

    return img

def h_append(img1, img2, put_line=False):
    """Appends two images horizontally, filling them with zeros to match 
        dimensions if needed."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h2 > h1:
        img1 = fill(img1, w1, h2)
    else:
        img2 = fill(img2, w2, h1)

    if put_line:
        line = img1.max()*np.ones(dtype=img1.dtype, shape=(img1.shape[0], 1))
        return np.hstack((img1, line, img2))
    return np.hstack((img1, img2))

def v_append(img1, img2, put_line=False):
    """Appends two images vertically, filling them with zeros to match 
        dimensions if needed."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if w2 > w1:
        img1 = fill(img1, w2, h1)
    else:
        img2 = fill(img2, w1, h2)

    if put_line:
        line = img1.max()*np.ones(dtype=img1.dtype, shape=(1, img1.shape[1]))
        return np.vstack((img1, line, img2))
    return np.vstack((img1, img2))

def append(img1, img2, hor=True):
    """Appends two images either horizontally or vertically."""
    return h_append(img1, img2) if hor else v_append(img1, img2)

def intensity_map(img, pyr_lvl=3, cs_ksizes=(3, 7),
    dtype=np.float32, debug=False):
    """Gets intensity map by by summing up center_surround on
        multiple scales, and kernel sizes.
        If debug is True, returns image with intermediate results."""
    #getting all kernels
    cs_kernels = [get_center_surround_kernel(ks, dtype) for ks in cs_ksizes]
    #initial value for intensity map
    im_img = np.zeros(dtype=dtype, shape=img.shape)
    #debug image
    debug_img = None

    #iterating over pyramid levels (from 0 to pyr_lvl)
    for i in xrange(pyr_lvl+1):
        #getting downsampled image
        img = cv2.pyrDown(img) if i > 0 else img
        #partial debug image
        db_img = None

        for k in cs_kernels:
            #getting center-surround image
            cs_img = center_surround(img, k)
            #updating debug partial image
            if debug:
                db_img = cs_img if db_img is None else h_append(db_img, cs_img)
            #rescaling into original image dimensions
            cs_img = rep_pyrUp(cs_img, i)
            #updating debug partial image
            if debug and i > 0:
                db_img = append(db_img, cs_img)
            #summing contribution
            im_img += cs_img
    
        #updating debug image
        if debug:
            debug_img = db_img if debug_img is None else \
                v_append(debug_img, db_img) 

    return debug_img, im_img

def str_dim(img):
    """Returns a string with image dimensions (height x width x depth)."""
    return " x ".join(str(dim) for dim in img.shape)

def str_type(img):
    """Returns a string with image type."""
    return str(img.dtype)

def str_info(img):
    """Returns a string with info for image."""
    return "dims: %s | type: %s" % (str_dim(img), str_type(img))

def scale(img, new_min=0.0, new_max=255.0):
    """Scales pixels of input image to new interval."""
    minn = img.min()
    maxx = img.max()
    sigma = maxx - minn
    out_img = new_min + (img - minn)/sigma*new_max

    return out_img

def display(img, title, to_uint8=True):
    """Displays image in viewable format with useful info."""
    if to_uint8:
        img = np.array(scale(img, 0, 255), dtype=np.uint8)
    cv2.imshow("'%s' (%s)" % (title, str_info(img)), img)

def save(img, name, to_uint8=True):
    """Saves image in viewable format with useful info."""
    if to_uint8:
        img = np.array(scale(img, 0, 255), dtype=np.uint8)
    cv2.imwrite(name, img)

def error(msg, code=1):
    """Prints error message and exits with code."""
    print "error:", msg
    exit(code)

def resize(img, max_w, max_h, scale=0.75):
    """Resizes image to fit dimensions."""
    h, w = img.shape[:2]
    while h > max_h or w > max_w:
        h, w = img.shape[:2]
        img = cv2.resize(img, tuple(map(int, (scale*w, scale*h))))

    return img

def str_to_list(string, tp, delim=","):
    """Divides string separated by delimiter and converts to tp."""
    return map(tp, string.split(delim))

def main():
    #command-line arguments
    img_file = oarg.Oarg("-i --img", "", "path to image", 0) 
    pyr_lvl = oarg.Oarg("-p --pyr-lvl", 3, "levels for pyramid") 
    #must be comma-separated
    str_cs_ksizes= oarg.Oarg("-c --cs-ks", "3,7", 
        "center-surround kernel sizes") 
    #must be comma-separated
    str_feats = oarg.Oarg("-f --features", "l1,l0,r,g,b,y", "features to use")
    max_w = oarg.Oarg("-W --max-w", 800, "maximum width for image")
    max_h = oarg.Oarg("-H --max-h", 600, "maximum hwight for image")
    debug = oarg.Oarg("-d --debug", False, "debug mode")
    save_dir = oarg.Oarg("-s --save-dir", "", "directory to save images")
    hlp = oarg.Oarg("-h --help", False, "this help message")

    #parsing args
    oarg.parse()

    #help message
    if hlp.val:
        oarg.describeArgs("options:", def_val=True)
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
    
    #resizing image
    img = resize(img, max_w.val, max_h.val)
    #adapting image dimensions for proper pyramid up/downscaling
    img = pyr_prepare(img, pyr_lvl.val)
    #converting image to LAB colorspace
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = (lab_img[:, :, i] for i in range(3))

    #displaying original image
    display(img, "original image", False)

    #getting center-surround kernel sizes
    cs_ksizes = str_to_list(str_cs_ksizes.val, int)
    feats = str_to_list(str_feats.val, str)

    ims = []
    #computing intensity maps
    for feat in feats:
        print "in %s ..." % feat
        input_img = get_lab_attr(img, feat)
        db_im, im = intensity_map(input_img, pyr_lvl.val, cs_ksizes,    
            debug=debug)
        
        #displaying images
        display(img, "input image %s" % feat)
        display(im, "intensity map %s" % feat)
        if debug:
            display(db_im, "intermediary intensity_map %s" % feat)

        #appending partial intensity map
        ims.append(im)

        #saving images
        if save_dir.val:
            save(im, "%s/%s_map.png" % (save_dir.val, feat))
            if debug:
                save(db_im, "%s/%s_db_map.png" % (save_dir.val, feat))

    #computing final intensity map
    final_im = sum(ims)
    
    #displaying final result
    display(final_im, "final intensity map")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #saving final result
    if save_dir.val:
        save(final_im, "%s/final_map.png" % save_dir.val)

if __name__ == "__main__":
    main()
