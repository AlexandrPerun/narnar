import cv2
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import optical_flow


def get_points_from_canvas(objects, color="blue"):
    objects = objects.filter(items=['left', 'top', 'fill', 'width', 'height', 'scaleX', 'scaleY'])
    objects.rename(columns={'left': 'x', 'top': 'y', 'fill': 'color', 'width': 'radiusX', 'height': 'radiusY'},
                   inplace=True)

    points = objects[objects.color.isin([color])]
    points.radiusX = points.radiusX.div(2)
    points.radiusY = points.radiusY.div(2)
    for i in range(len(points)):
        points.radiusX[i] *= points.scaleX[i]
        points.radiusY[i] *= points.scaleY[i]
        points.x[i] += points.radiusX[i]

    points.x = points.x.astype(int)
    points.y = points.y.astype(int)
    points.radiusX = points.radiusX.astype(int)
    points.radiusY = points.radiusY.astype(int)
    # st.dataframe(points)

    return [(points.x[i], points.y[i]) for i in points.index], [(points.radiusX[i], points.radiusY[i]) for i in points.index]


st.set_page_config(layout="wide", page_title="narnar")
drawing_mode = st.sidebar.selectbox(
    "Drawing mode:", ("drawing", "editing")
)
drawing_mode = 'point' if drawing_mode == "drawing" else "transform"
# color = st.sidebar.selectbox("Color:", ("blue", "red"))
color = "blue"
point_display_radius = st.sidebar.slider("Point display radius: ", 1, 5, 3)
arrows = st.sidebar.checkbox('Draw arrows')
if arrows:
    arrows_thickness = st.sidebar.slider("Arrows thickness: ", 1, 5, 2)

im1 = st.sidebar.file_uploader("First image:", type=["png", "jpg"])
im2 = st.sidebar.file_uploader("Second image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
if im1:
    im1 = Image.open(im1)
    original_w, original_h = im1.size
    im1.thumbnail((600, 600), Image.ANTIALIAS)
    w, h = im1.size
    canvas_result = st_canvas(
        fill_color=color,
        stroke_width=0,
        stroke_color='blue',
        background_color='white',
        background_image=im1,
        update_streamlit=realtime_update,
        height=h,
        width=w,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius,
        key="canvas",
    )

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")

        if len(objects):
            col1, col2 = st.columns((2,1))
            with col2:
                with st.expander("Set Gunnar-Farneback algorithm parameters"):
                    pyr_scale = st.slider("pyr_scale: ", min_value=0.0, max_value=0.9, value=0.5, step=0.1)
                    levels = st.slider("levels: ", min_value=1, max_value=15, value=3, step=1)
                    winsize = st.slider("winsize: ", min_value=3, max_value=30, value=15, step=1)
                    iteration = st.slider("iteration: ", min_value=1, max_value=10, value=3, step=1)
                    poly_n = st.slider("poly_n: ", min_value=5, max_value=7, value=5, step=2)
                    poly_sigma = st.slider("poly_sigma: ", min_value=1.1, max_value=1.5, value=1.2, step=0.1)
                    flags = st.slider("flags: ", min_value=0, max_value=1, value=0, step=1)

            im2 = Image.open(im2)
            im2.thumbnail((600, 600), Image.ANTIALIAS)
            im1_gray, im2_gray = ImageOps.grayscale(im1), ImageOps.grayscale(im2)
            im1, im2, im1_gray, im2_gray = np.array(im1), np.array(im2), np.array(im1_gray), np.array(im2_gray)
            im1_gray, im2_gray = im1_gray.transpose(), im2_gray.transpose()
            points, radiuses = get_points_from_canvas(objects)

            config = optical_flow.Config(flow=None,
                                         pyr_scale=pyr_scale,
                                         levels=levels,
                                         winsize=winsize,
                                         iterations=iteration,
                                         poly_n=poly_n,
                                         poly_sigma=poly_sigma,
                                         flags=flags)
            result_points = optical_flow.opticalFlowFewPoints(points, im1_gray, im2_gray, config)

            if len(im2.shape) > 2:
                image = cv2.cvtColor(im2, cv2.COLOR_RGBA2RGB)
            else:
                image = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)

            for i in range(len(points)):
                image = cv2.ellipse(img=image, center=points[i], axes=radiuses[i],
                                    angle=0, startAngle=0, endAngle=360, color=(0, 0, 255), thickness=-1)
                image = cv2.ellipse(img=image, center=result_points[i], axes=radiuses[i],
                                    angle=0, startAngle=0, endAngle=360, color=(255, 0, 0), thickness=-1)
                if arrows:
                    image = cv2.arrowedLine(image, points[i], result_points[i],
                                            color=(0, 255, 0), thickness=arrows_thickness, tipLength=0.33)

            with col1:
                st.image(image)

                image = cv2.resize(image, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                _, image = cv2.imencode(".jpg", image)
                byte_im = image.tobytes()
                st.download_button(
                    label="Download result image",
                    data=byte_im,
                    file_name='result.jpg',
                    mime='image/jpeg',
                )
