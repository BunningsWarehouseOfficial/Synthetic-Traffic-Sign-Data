class SynthImage:
    def __init__(self, fg_path, class_num, damage_type=None, damage_tag=None, damage_ratio=0.0,
            transform_type=None, man_type=None, bg_path=None, bounding_axes=None):
        self.__check_class(class_num)
        self.__check_damage(damage_ratio)

        self.fg_path = fg_path
        self.class_num = class_num

        self.damage_type = damage_type
        self.damage_tag = damage_tag  #TODO: Turn into dictionary?
        self.damage_ratio = damage_ratio

        self.transform_type = transform_type
        self.man_type = man_type
        self.bg_path = bg_path

        self.bounding_axes = bounding_axes

    def __repr__(self):
        return f"fg_path={self.fg_path}"


    def set_damage(self, fg_path, damage_type, damage_tag, damage_ratio):
        self.__check_damage(damage_ratio)
        self.damage_type = damage_type
        self.damage_tag = damage_tag
        self.damage_ratio = damage_ratio

    def set_transformation(self, transform_type):
        self.__check_transformation(transform_type)
        self.transform_type = transform_type

    def set_manipulation(self, man_type):
        self.__check_manipulation(man_type)
        self.man_type = man_type

    def clone(self):
        return SynthImage(
            self.fg_path,
            self.class_num,
            self.damage_type,
            self.damage_tag,
            self.damage_ratio,
            self.bg_path
        )
        
    def write_label(self, labels_file, labels_dict, labels_format, index, img_path, img_dims):
        axes = self.bounding_axes
        bounds = f"{axes[0]} {axes[1]} {axes[2]} {axes[3]}"
        if labels_format == "retinanet":
            labels_file.write(f"{self.fg_path} {bounds} class={self.class_num} "
                        f"{self.damage_type}={self.damage_tag} damage={self.damage_ratio} "
                        f"transform_type={self.transform_type} man_type={self.man_type} "
                        f"bg={self.bg_path}\n")
        elif labels_format == "coco":
            labels_dict['annotations'].append({
                "id": index,
                "image_id": index,
                "category_id": self.class_num,
                "bbox": [axes[0], axes[2], axes[1] - axes[0], axes[3] - axes[2]],
                "area": (axes[1] - axes[0]) * (axes[3] - axes[2]),
                "segmentation": [],
                "iscrowd": 0
            })
            from datetime import datetime
            labels_dict['images'].append({
                "id": index,
                "license": 1,
                "file_name": img_path,
                "height": img_dims[0],
                "width": img_dims[1],
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    def __check_class(self, class_num):
        if class_num < 0:
            raise TypeError(f"class_num={class_num} is invalid: must be >= 0")

    def __check_damage(self, damage_ratio):
        if damage_ratio < 0.0 or damage_ratio > 1.0:
            raise TypeError(f"damage_ratio={damage_ratio} is invalid: must have 0.0 <= damage_ratio <= 1.0")

    def __check_transformation(self, transform_type):
        if transform_type < 0:
            raise TypeError(f"transform_type={transform_type} is invalid: must be >= 0") #TODO: Check

    def __check_manipulation(self, man_type):
        if man_type is None:
            raise TypeError(f"man_type={man_type} is invalid: must be valid string")
