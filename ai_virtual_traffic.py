import cv2
import numpy as np
import tensorflow as tf

# Load model
model_path = 'ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'

# Load video
cap = cv2.VideoCapture(r'C:\Users\dhars\OneDrive\Documentos\Desktop\AI_Virtual_Traffic\traffic_video.mp4.mkv')




# Label map
LABEL_MAP = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    6: 'bus', 7: 'train', 8: 'truck'
}

# Load the frozen model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        # Tensors
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Expand and detect
            input_frame = np.expand_dims(frame, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [boxes_tensor, scores_tensor, classes_tensor, num_detections],
                feed_dict={image_tensor: input_frame})

            h, w, _ = frame.shape

            # Draw results
            for i in range(int(num[0])):
                if scores[0][i] > 0.5:
                    class_id = int(classes[0][i])
                    box = boxes[0][i] * np.array([h, w, h, w])
                    (ymin, xmin, ymax, xmax) = box.astype(int)
                    label = LABEL_MAP.get(class_id, 'Unknown')
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Traffic Detection', frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
