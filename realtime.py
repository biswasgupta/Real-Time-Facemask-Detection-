import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
from collections import Counter
from ultralytics import YOLO
import urllib.request

def download_dnn_files():
    files = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }

    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
    print("DNN files are ready.")

download_dnn_files()

class FaceMaskDetector:
    def __init__(
        self,
        model_dir="saved_models",
        yolo_model_path="yolo_v8.pt",
        model_choice="ensemble",
    ):
        """
        Initialize the Face Mask Detector with necessary models.
        model_choice must be one of ["resnet50", "resnet18", "yolo", "ensemble"].
        """
        self.model_dir = model_dir
        self.yolo_model_path = yolo_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_choice = model_choice.lower().strip()
        assert self.model_choice in ["resnet50", "resnet18", "yolo", "ensemble"], \
            "model_choice must be 'resnet50', 'resnet18', 'yolo', or 'ensemble'."

        print(f"Using device: {self.device}")
        print(f"Loading models for mode: {self.model_choice}")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # -------------------------
        # Load YOLO model if requested or for ensemble
        # -------------------------
        self.yolo_available = False
        if self.model_choice == "yolo" or self.model_choice == "ensemble":
            try:
                self.yolo_model = YOLO(yolo_model_path)
                print("YOLO model loaded successfully")
                self.yolo_available = True
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                print("Continuing without YOLO for now...")
                self.yolo_available = False

        # -------------------------
        # Load ResNet-50 and ResNet-18 if needed
        # -------------------------
        # We only load them if the choice is resnet50, resnet18, or ensemble
        self.resnet50_model = None
        self.resnet18_model = None

        if self.model_choice in ["resnet50", "ensemble"]:
            self.resnet50_model = self._load_model(
                model_path=f"{model_dir}/model_resnet50_state_dict.pth",
                model_type='resnet50'
            )
        
        if self.model_choice in ["resnet18", "ensemble"]:
            self.resnet18_model = self._load_model(
                model_path=f"{model_dir}/model_resnet18_state_dict.pth",
                model_type='resnet18'
            )
        
        # -------------------------
        # Download face detection model if needed
        # (for resnets or ensemble, we need face detection bounding boxes)
        # -------------------------
        if self.model_choice != "yolo":
            self._download_dnn_models()
            self.face_detector = cv2.dnn.readNetFromCaffe(
                "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"
            )
        else:
            self.face_detector = None  # Not needed if YOLO-only

        print("Initialization complete!")

    def _create_model(self, model_name='resnet50', num_classes=3, pretrained=True):
        """Create a model with improved architecture."""
        if model_name == 'resnet50':
            if pretrained:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                model = models.resnet50(weights=None)
        elif model_name == 'resnet18':
            if pretrained:
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                model = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace the classifier head
        if model_name.startswith('resnet'):
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )
        
        return model

    def _load_model(self, model_path, model_type='resnet18'):
        """Load a trained model."""
        try:
            model = self._create_model(model_name=model_type, pretrained=False, num_classes=3)
            if not os.path.isfile(model_path):
                print(f"Warning: Model file {model_path} not found")
                print(f"Using {model_type} with random weights (then pretrained ImageNet).")
                model = self._create_model(model_name=model_type, pretrained=True, num_classes=3)
            else:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"{model_type} loaded successfully from {model_path}")

            model.eval()
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            print(f"Using {model_type} with ImageNet weights instead")
            model = self._create_model(model_name=model_type, pretrained=True, num_classes=3)
            model.eval()
            model.to(self.device)
            return model

    def _download_dnn_models(self):
        """Download DNN face detector model files if they don't exist."""
        model_file = "res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "deploy.prototxt"
        
        if not os.path.isfile(model_file):
            print(f"Downloading {model_file}...")
            os.system("wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
        
        if not os.path.isfile(config_file):
            print(f"Downloading {config_file}...")
            os.system("wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt")
        
        if os.path.isfile(model_file) and os.path.isfile(config_file):
            print("DNN model files are ready.")
            return True
        else:
            print("Failed to download DNN model files. Please download them manually:")
            print("wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt")
            print("wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
            return False

    def predict_mask(self, face_image, model):
        """Predict mask status from a face image using a given ResNet model."""
        transform = transforms.Compose([
            transforms.Resize((226, 226)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(face_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        prediction_idx = predicted.item()
        probability = probabilities[0][prediction_idx].item()
        
        class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        predicted_class = class_names[prediction_idx]
        
        return predicted_class, probability

    def ensemble_predict(self, face_image, yolo_class=None):
        """Predict class using ensemble voting from ResNet-50, ResNet-18, and optional YOLO."""
        predictions = []
        
        if self.resnet50_model is not None:
            pred_resnet50, _ = self.predict_mask(face_image, self.resnet50_model)
            predictions.append(pred_resnet50)
        
        if self.resnet18_model is not None:
            pred_resnet18, _ = self.predict_mask(face_image, self.resnet18_model)
            predictions.append(pred_resnet18)
        
        # Add YOLO classification if available
        if yolo_class is not None:
            predictions.append(yolo_class)
        
        vote_counts = Counter(predictions)
        top_votes = vote_counts.most_common()

        # If there's a tie and YOLO is involved, YOLO wins
        if len(top_votes) > 1 and top_votes[0][1] == top_votes[1][1] and yolo_class:
            final_prediction = yolo_class
        else:
            final_prediction = top_votes[0][0]

        return final_prediction

    def _process_frame_yolo(self, frame):
        """
        Process a single frame in YOLO-only mode:
        We rely purely on YOLO's bounding boxes and classes for face-mask classification.
        """
        if not self.yolo_available:
            cv2.putText(frame, "YOLO model not available.", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        results = self.yolo_model(frame, verbose=False)
        detections = results[0].boxes  # bounding boxes

        if len(detections) == 0:
            # No objects detected
            cv2.putText(frame, "No detections (YOLO).", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        for det in detections:
            # det.xyxy[0] => [x1, y1, x2, y2], det.cls => class index
            box = det.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            cls_index = int(det.cls)
            # YOLO class name
            predicted_class = self.yolo_model.names[cls_index]

            # Color logic
            if predicted_class == 'with_mask':
                color = (0, 255, 0)
            elif predicted_class == 'without_mask':
                color = (0, 0, 255)
            elif predicted_class == 'mask_weared_incorrect':
                color = (255, 0, 0)
            else:
                color = (255, 255, 255)  # If YOLO has some other class

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{predicted_class}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    def _process_frame_caffe(self, frame):
        """
        Process a single frame using the Caffe face detector for bounding boxes,
        then do classification with the chosen ResNet(s) or ensemble of ResNets+YOLO.
        """
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        # If YOLO is also used in ensemble, get YOLO's raw predictions
        yolo_predictions = []
        if self.model_choice == "ensemble" and self.yolo_available:
            try:
                yolo_results = self.yolo_model(frame, verbose=False)
                yolo_predictions = [self.yolo_model.names[int(det.cls)] 
                                    for det in yolo_results[0].boxes]
            except Exception as e:
                print(f"Error with YOLO prediction: {e}")
                yolo_predictions = []

        face_count = 0
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                face_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)

                face_roi = frame[startY:endY, startX:endX]
                if face_roi.size == 0:
                    continue

                try:
                    face_image = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    print(f"Error converting to PIL Image: {str(e)}")
                    continue

                # If we're in ensemble mode, we get YOLO label (if any) by index
                yolo_class = None
                if self.model_choice == "ensemble" and yolo_predictions and face_count <= len(yolo_predictions):
                    yolo_class = yolo_predictions[face_count-1]

                # If using a single ResNet model
                if self.model_choice == "resnet50" and self.resnet50_model is not None:
                    predicted_class, _ = self.predict_mask(face_image, self.resnet50_model)
                elif self.model_choice == "resnet18" and self.resnet18_model is not None:
                    predicted_class, _ = self.predict_mask(face_image, self.resnet18_model)
                elif self.model_choice == "ensemble":
                    # Use combined logic
                    predicted_class = self.ensemble_predict(face_image, yolo_class)
                else:
                    # If there's no valid model in this path, skip
                    continue

                # Color based on prediction
                if predicted_class == 'with_mask':
                    color = (0, 255, 0)
                elif predicted_class == 'without_mask':
                    color = (0, 0, 255)
                elif predicted_class == 'mask_weared_incorrect':
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)  # fallback

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                label = f"{predicted_class}"
                y_text = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if face_count == 0:
            cv2.putText(frame, "No faces detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def process_image(self, image_path, show_result=True):
        """Process an image from file path."""
        if not os.path.isfile(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return None

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at path: {image_path}")
            return None

        if self.model_choice == "yolo":
            # YOLO-only mode
            result_img = self._process_frame_yolo(img)
        else:
            # ResNet or ensemble mode
            result_img = self._process_frame_caffe(img)

        if show_result:
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(result_rgb)
            plt.title("Face Mask Detection Results")
            plt.axis('off')
            plt.show()

        return result_img

    def start_webcam(self):
        """Start real-time face mask detection using webcam."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam. Make sure your camera is connected.")
            return

        print("Starting webcam detection. Press 'q' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture video feed.")
                break

            if self.model_choice == "yolo":
                frame = self._process_frame_yolo(frame)
            else:
                frame = self._process_frame_caffe(frame)

            cv2.imshow('Face Mask Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_interface(self):
        """Run a simple command-line interface."""
        while True:
            print("\n=== Face Mask Detection System ===")
            print("1. Real-time detection (webcam)")
            print("2. Process an image file")
            print("3. Change model choice (current: {})".format(self.model_choice))
            print("4. Exit")

            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '1':
                self.start_webcam()
            elif choice == '2':
                image_path = input("Enter the path to the image file: ")
                try:
                    self.process_image(image_path, show_result=True)
                except Exception as e:
                    print(f"Error processing image: {e}")
            elif choice == '3':
                print("Possible choices: 'resnet50', 'resnet18', 'yolo', 'ensemble'")
                new_choice = input("Enter new model choice: ").lower().strip()
                if new_choice in ["resnet50", "resnet18", "yolo", "ensemble"]:
                    self.model_choice = new_choice
                    print(f"Model choice updated to: {self.model_choice}")
                else:
                    print("Invalid choice. Please try again.")
            elif choice == '4':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    # Initialize the detector. You can change 'model_choice' here:
    detector = FaceMaskDetector(
        model_dir="saved_models",      # Directory where model files are stored
        yolo_model_path="yolo_v8.pt",  # Path to YOLO model
        model_choice="ensemble"        # One of "resnet50", "resnet18", "yolo", "ensemble"
    )
    
    # Start the interface
    detector.run_interface()
