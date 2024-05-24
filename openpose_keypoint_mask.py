import numpy as np
import torch
import math
from PIL import Image, ImageDraw
from nodes import MAX_RESOLUTION

def draw_square(draw, center_x, center_y, size, aspect_ratio, color):
    aspect_ratio = aspect_ratio if aspect_ratio !=0 else 1
    draw.rectangle([(center_x - size / 2, center_y - size / 2 / aspect_ratio),
                    (center_x + size / 2, center_y + size / 2 / aspect_ratio)], fill=color)

def draw_triangle(draw, center_x, center_y, size, aspect_ratio, color):
    aspect_ratio = aspect_ratio if aspect_ratio !=0 else 1
    draw.polygon([(center_x, center_y - size / 2 / aspect_ratio),
                  (center_x + size / 2, center_y + size / 2 / aspect_ratio),
                  (center_x - size / 2, center_y + size / 2/ aspect_ratio)], fill=color)
def draw_oval(draw, center_x, center_y, size, aspect_ratio, color):
    aspect_ratio = aspect_ratio if aspect_ratio !=0 else 1
    draw.ellipse([(center_x - size / 2, center_y - size / 2 / aspect_ratio),
                  (center_x + size / 2, center_y + size / 2 / aspect_ratio)], fill=color)

def rotate_points(points, center=(0, 0), angle_deg=0):  
    """  
    Rotate a list of 2D points around a center point by a given angle (in degrees).  
      
    :param points: List of tuples representing 2D points (e.g., [(x1, y1), (x2, y2), ...])  
    :param center: Tuple representing the center point to rotate around (default is (0, 0))  
    :param angle_deg: Angle in degrees to rotate the points  
    :return: List of rotated 2D points  
    """  
    angle_rad = math.radians(angle_deg)  # Convert degrees to radians  
    cx, cy = center  # Extract center coordinates  
    rotated_points = []  
      
    for x, y in points:  
        # Translate point back to origin  
        ox = x - cx  
        oy = y - cy  
          
        # Perform rotation  
        nx = ox * math.cos(angle_rad) - oy * math.sin(angle_rad)  
        ny = ox * math.sin(angle_rad) + oy * math.cos(angle_rad)  
          
        # Translate point back to original position  
        x_rotated = nx + cx  
        y_rotated = ny + cy  
          
        # Append rotated point to the list  
        rotated_points.append((x_rotated, y_rotated))  
      
    return rotated_points
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0) 

class OpenPoseKeyPointMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        shapes = ["oval","square","triangle"]
        modes = ["box","torso"]
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "image_width": ("INT", { "min": 0, "max": MAX_RESOLUTION }),
                "image_height": ("INT", { "min": 0, "max": MAX_RESOLUTION }),
            },
            "optional": {
                "points_list": ("STRING", {"multiline": True, "default": "1,8,11"}),
                "mode": (modes,{"default": "box"}),
                "shape": (shapes,{"default": "oval"}),
                "x_offset": ("FLOAT", { "min": -10, "max": 10, "default": 0.0 }),
                "y_offset": ("FLOAT", { "min": -10, "max": 10, "default": 0.0 }),
                "x_zoom": ("FLOAT", { "min": 0, "max": 100, "default": 1.0 }),
                "y_zoom": ("FLOAT", { "min": 0, "max": 100, "default": 1.0 }),
                "person_index": ("INT", { "default": -1 }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "mask_keypoints"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def min_area_rectangle(self,points):
        min_x=MAX_RESOLUTION
        min_y=MAX_RESOLUTION
        max_x=0
        max_y=0
        for p in points:
            if p[0]<min_x:
                min_x=p[0]
            if p[1]<min_y:
                min_y=p[1]
            if p[0]>max_x:
                max_x=p[0]
            if p[1]>max_y:
                max_y=p[1]
        return (min_x, min_y, max_x, max_y)
    def get_keypoint_from_list(self,list, item,pose):
        idx_x = item*3
        idx_y = idx_x + 1
        idx_conf = idx_y + 1
        x=list[idx_x]
        y=list[idx_y]
        z=list[idx_conf]
        if x >= 1.0 or y >= 1.0:
            x=x/pose["canvas_width"]
            y=y/pose["canvas_height"]
        return (x,y,z)
    def box_keypoint(self, pose, points_we_want, person_number=0):
        if person_number >= len(pose["people"]):
            return (0,0,0,0,0)
        min_size=10
        canvas_width =pose["canvas_width"]
        canvas_height =pose["canvas_height"]
        points=[]
        for element in points_we_want:
            (x,y,z) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], element,pose)
            if z != 0.0:
                points.append((x*canvas_width,y*canvas_height))
        
        if len(points) > 0:
            if len(points) < 3:
                points.append((points[0][0]+min_size,points[0][1]))
            if len(points) < 3:
                points.append((points[0][0],points[0][1]+min_size))
        else:
            return (0,0,0,0,0)
        rectangle = self.min_area_rectangle(points)
        width_rc  = abs(rectangle[0]-rectangle[2])
        height_rc = abs(rectangle[1]-rectangle[3])
        x = rectangle[0]
        y = rectangle[1]
        angle=0
        center=(0,0)
        for i in range(-90,90,1):
            if i==0:
                continue
            rectangle = self.min_area_rectangle(rotate_points(points,center=center,angle_deg=i))
            area = abs(rectangle[0]-rectangle[2])*abs(rectangle[1]-rectangle[3])
            if area < width_rc*height_rc:
                width_rc  = abs(rectangle[0]-rectangle[2])
                height_rc = abs(rectangle[1]-rectangle[3])
                x = rectangle[0]
                y = rectangle[1]
                angle=i
        x,y=rotate_points([[x,y]],center=center,angle_deg=(0-angle))[0]
        print(x,y,angle,height_rc,width_rc,math.sin(math.radians(angle)))
        return (x/canvas_width,y/canvas_height,width_rc/canvas_width,height_rc/canvas_height,angle)
    def get_torso_width(self,pose,person_number=0):
        if person_number >= len(pose["people"]):
            return 0
        (x,y,z) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 1,pose)
        (x2,y2,z2) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 2,pose)
        if z2==0.0:
            (x2,y2,z2) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 5,pose)
        if z != 0.0 and z2 != 0.0:
            return abs(x-x2)*2
        return 0
    def get_torso_height(self,pose,person_number=0):
        if person_number >= len(pose["people"]):
            return 0
        (x,y,z) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 1,pose)
        (x2,y2,z2) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 8,pose)
        if z2==0.0:
            (x2,y2,z2) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 11,pose)
        if z != 0.0 and z2 != 0.0:
            return abs(y-y2)
        return 0
    def get_torso_angle(self,pose,person_number=0):
        if person_number >= len(pose["people"]):
            return 0
        (x,y,z) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 1,pose)
        (x2,y2,z2) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 2,pose)
        (x3,y3,z3) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 5,pose)
        if z != 0.0 and z2 != 0.0 and z3 != 0.0:
            return math.degrees(math.atan2((y3+y2)/2 - y,(x3+x2)/2 - x))
        return 0
    def get_head_width(self,pose,person_number=0):
        if person_number >= len(pose["people"]):
            return 0
        (x,y,z) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 0,pose)
        (x2,y2,z2) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 14,pose)
        (x3,y3,z3) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 17,pose)
        if x3>x2:
            x2=x3
        if z != 0.0 and z2 != 0.0:
            return abs(x-x2)*2
        return 0
    def get_head_height(self,pose,person_number=0):
        if person_number >= len(pose["people"]):
            return 0
        (x,y,z) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 0,pose)
        (x2,y2,z2) = self.get_keypoint_from_list(pose["people"][person_number]["pose_keypoints_2d"], 1,pose)
        if z != 0.0 and z2 != 0.0:
            return abs(y-y2)
        return 0
    def make_shape(self, width, height, rotation,shape,x_offset=0, y_offset=0, zoom=1.0,):
        bg_color = (0,0,0)
        shape_color = (255,255,255)
        if width==0:
            width=1
        if height==0:
            height=1
        back_img = Image.new("RGB", (width, height), color=bg_color)
        shape_img = Image.new("RGB", (width, height), color=shape_color)
        shape_mask = Image.new('L', (width, height))
        draw = ImageDraw.Draw(shape_mask)   

        center_x = width // 2 + x_offset
        center_y = height // 2 + y_offset         
        size = min(width - x_offset, height - y_offset) * zoom
        aspect_ratio = (width / height) if height != 0 else 1
        if aspect_ratio > 1:
            size = max(width - x_offset, height - y_offset) * zoom
        color = 'white'

        shape_functions = {
            'oval': draw_oval,
            'square': draw_square,
            'triangle': draw_triangle,
        }

        if shape in shape_functions:
            shape_function = shape_functions.get(shape)
            shape_function(draw, center_x, center_y, size, aspect_ratio, color)
        if shape == "diagonal regions":
            draw.polygon([(width, 0), (width, height), (0, height)], fill=color)

        shape_mask = shape_mask.rotate(rotation, center=(center_x, center_y))
        result_image = Image.composite(shape_img, back_img, shape_mask) 
        return result_image
    def mask_keypoints(self,pose_keypoint, image_width, image_height, points_list="1,8,11",
                       mode="box",shape="oval",x_offset=0, y_offset=0, x_zoom=1.0, y_zoom=1.0,
                       person_index=-1):
        points_we_want = []
        for element in points_list.split(","):
            if element.isdigit():
                points_we_want.append(int(element))
        full_masks = []
        for pose in pose_keypoint:
            out_img = Image.new("RGB", (image_width, image_height))
            for person_number in range(len(pose["people"])):
                if person_index>=0 and person_index!=person_number:
                    continue
                if mode == "torso":
                    box=self.box_keypoint(pose, [1], person_number)
                    point_width=self.get_torso_width(pose,person_number)
                    point_height=self.get_torso_height(pose,person_number)
                    if point_width==0:
                        point_width=self.get_head_width(pose,person_number)*2
                    if point_height==0:
                        point_height=self.get_head_height(pose,person_number)*2
                    out_img_x=int((box[0]-point_width/2)*image_width)
                    out_img_y=int(box[1]*image_height)
                    out_x_offset=x_offset*point_width*x_zoom*image_width
                    out_y_offset=y_offset*point_height*y_zoom*image_height
                    shape_img=self.make_shape(int(point_width*image_width*x_zoom),int(point_height*image_height*y_zoom),0,shape)
                    rotation=self.get_torso_angle(pose,person_number)
                    shape_img = shape_img.rotate(rotation,expand=True)
                else:
                    box=self.box_keypoint(pose, points_we_want, person_number)
                    out_img_x=int(box[0]*image_width)
                    out_img_y=int(box[1]*image_height)
                    box_width=int(box[2]*image_width)
                    box_height=int(box[3]*image_height)
                    out_x_offset=x_offset*box_width*x_zoom
                    out_y_offset=y_offset*box_height*y_zoom
                    shape_img=self.make_shape(int(box_width*x_zoom),int(box_height*y_zoom),0,shape)
                if len(box)==5:
                    rotation=box[4]
                    shape_img = shape_img.rotate(rotation,expand=True)
                    if rotation<0:
                        out_x_offset=out_x_offset+int(box_height*x_zoom*math.sin(math.radians(rotation)))
                    if rotation>0:
                        out_y_offset=out_y_offset-int(box_width*y_zoom*math.sin(math.radians(rotation)))
                out_img.paste(shape_img,(int(out_img_x+out_x_offset), int(out_img_y+out_y_offset)))
            mask=out_img.convert("L")
            # out_img.show()
            full_masks.append(pil2tensor(mask))
        return (torch.cat(full_masks, dim=0),)
NODE_CLASS_MAPPINGS = {
    "Openpose Keypoint Mask": OpenPoseKeyPointMask,
}
