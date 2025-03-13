import { useState, useEffect } from "react";
import { Upload } from "lucide-react";
import axios from "axios";

function UploadImageTab() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [outputImage1, setOutputImage1] = useState<string | null>(null); // Output from Model 1
  const [outputImage2, setOutputImage2] = useState<string | null>(null); // Output from Model 2
  const [featureImage, setFeatureImage] = useState<string | null>(null); // Feature maps
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    if (loading) {
      console.log("Processing image ... ");
    }
  }, [loading]);

  // handlers
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    console.log("?");
    setSelectedFile(file);
    if (file) sendToBackend(file);
  };

  const sendToBackend = async (file: File) => {
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://localhost:8000/process-image",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (response.data.image_url_model1 && response.data.image_url_model2) {
        setOutputImage1(response.data.image_url_model1); // Set output from Model 1
        setOutputImage2(response.data.image_url_model2); // Set output from Model 2
        setFeatureImage(response.data.feature_maps_url); // Set feature maps
      }
    } catch (error) {
      console.error("Error uploading file:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteImage = async () => {
    setSelectedFile(null);
    setOutputImage1(null);
    setOutputImage2(null);
    setFeatureImage(null);
  };

  return (
    <div style={styles.container}>
      {/* Row for the three square containers */}
      <div style={styles.row}>
        {/* Left Container - Upload */}
        <div style={styles.squareContainer}>
          {selectedFile ? (
            <div style={styles.imageWrapper}>
              <img
                src={URL.createObjectURL(selectedFile)}
                style={styles.image}
                alt="Uploaded"
              />
              <button
                onClick={handleDeleteImage}
                style={styles.deleteButton}
                title="Delete Image"
              >
                X
              </button>
            </div>
          ) : (
            <label htmlFor="fileUpload" style={styles.uploadLabel}>
              <Upload size={32} style={styles.uploadIcon} />
              <span style={styles.uploadText}>Upload Image</span>
            </label>
          )}
          <input
            type="file"
            onChange={handleFileUpload}
            style={{ display: "none" }}
            id="fileUpload"
          />
        </div>

        {/* Middle Container - Processed Image from Model 1 */}
        <div style={styles.squareContainer}>
          {loading ? (
            <p style={styles.placeholderText}>Processing...</p>
          ) : outputImage1 ? (
            <div style={styles.imageWrapper}>
              <img
                src={outputImage1}
                style={styles.image}
                alt="Processed Output (Model 1)"
              />
            </div>
          ) : (
            <p style={styles.placeholderText}>Nothing here...yet!</p>
          )}
        </div>

        {/* Right Container - Processed Image from Model 2 */}
        <div style={styles.squareContainer}>
          {loading ? (
            <p style={styles.placeholderText}>Processing...</p>
          ) : outputImage2 ? (
            <div style={styles.imageWrapper}>
              <img
                src={outputImage2}
                style={styles.image}
                alt="Processed Output (Model 2)"
              />
            </div>
          ) : (
            <p style={styles.placeholderText}>Nothing here...yet!</p>
          )}
        </div>
      </div>

      {/* Feature Map Container - Below the squares */}
      <div style={styles.featureImageWrapper}>
        {featureImage ? (
          <img
            src={featureImage}
            style={styles.featureImage}
            alt="Processed Feature Map"
          />
        ) : (
          <p style={styles.placeholderText}>Feature map placeholder</p>
        )}
      </div>
    </div>
  );
}

// Internal CSS styles
const styles = {
  container: {
    display: "flex",
    flexDirection: "column", // Stack children vertically
    gap: "1.5rem",
    width: "100%",
  },
  row: {
    display: "flex",
    gap: "1.5rem", // Space between the three squares
    width: "100%",
  },
  squareContainer: {
    width: "512px",
    height: "512px",
    border: "1px solid #e5e7eb",
    borderRadius: "0.5rem",
    backgroundColor: "#f9fafb",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  featureImageWrapper: {
    width: "calc(1124px + 2.0rem)",
    height: "256px",
    border: "1px solid #e5e7eb",
    borderRadius: "0.5rem",
    backgroundColor: "#f9fafb",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  featureImage: {
    width: "100%",
    height: "100%",
    borderRadius: "0.5rem",
    objectFit: "contain",
  },
  uploadLabel: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    cursor: "pointer",
  },
  uploadIcon: {
    color: "#6b7280",
  },
  uploadText: {
    color: "#6b7280",
  },
  image: {
    width: "100%",
    height: "100%",
    borderRadius: "0.5rem",
  },
  placeholderText: {
    color: "#6b7280",
  },
  imageWrapper: {
    width: "100%",
    height: "100%",
  },
  deleteButton: {
    position: "absolute",
    top: "8px",
    right: "8px",
    backgroundColor: "#f87171",
    color: "#fff",
    border: "none",
    borderRadius: "50%",
    width: "30px",
    height: "30px",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    cursor: "pointer",
  },
};

export default UploadImageTab;
