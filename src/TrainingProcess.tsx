import { useState, useEffect } from "react";
import { Upload } from "lucide-react";
import axios from "axios";

function TrainingProcessTab() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [lowImage, setLowImage] = useState<string | null>(null);
  const [highImage, setHighImage] = useState<string | null>(null);
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
        "http://localhost:8000/noisy-image",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (response.data.high_snr_url && response.data.low_snr_url) {
        setHighImage(response.data.high_snr_url);
        setLowImage(response.data.low_snr_url);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteImage = async () => {
    setSelectedFile(null);
    setLowImage(null);
    setHighImage(null);
  };

  return (
    <div style={styles.container}>
      {/* Container 1: Google Drive Link for Downsized Images */}
      <div style={styles.squareContainer}>
        <h3 style={styles.heading}>Downsized Images</h3>
        <p style={styles.description}>
          Download the downsized images from{" "}
          <a
            href="https://drive.google.com/drive/folders/1UbJ1J8XsUafgEksaAiGD2MTiySAEsaPs"
            target="_blank"
            rel="noopener noreferrer"
            style={styles.link}
          >
            Google Drive
          </a>
          . <br></br> <br></br> These images are resized to 512 * 512 pixels and
          ready for training. We used a split of ? for training and evaluation
          set.
        </p>
      </div>

      {/* Container 2: Adding Noise for Training (Demo) */}
      <div style={styles.squareContainer}>
        <h3 style={styles.heading}>Add Noise for Training</h3>
        <div style={styles.noiseContainer}>
          {/* Original Image */}
          <div style={styles.noiseBox}>
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

          {/* High SNR Image */}
          <div style={styles.noiseBox}>
            {highImage ? (
              <img src={highImage} style={styles.image} alt="High SNR" />
            ) : (
              <div style={styles.placeholderBox}>
                <p style={styles.placeholderText}> High SNR </p>
              </div>
            )}
            {/* <p style={styles.noiseLabel}>High SNR</p> */}
          </div>

          {/* Low SNR Image */}
          <div style={styles.noiseBox}>
            {lowImage ? (
              <img src={lowImage} style={styles.image} alt="Low SNR" />
            ) : (
              <div style={styles.placeholderBox}>
                <p style={styles.placeholderText}> Low SNR</p>
              </div>
            )}
            {/* <p style={styles.noiseLabel}>Low SNR</p> */}
          </div>
        </div>
      </div>

      {/* Container 3: CNN Diagram */}
      <div style={styles.squareContainer}>
        <h3 style={styles.heading}>CNN Architecture</h3>
        <img
          src="src/assets/CNN-diagram.png"
          style={styles.cnnImage}
          alt="CNN Diagram"
        />
        <p style={styles.description}>
          This diagram illustrates the architecture of the CNN used for
          training.
        </p>
      </div>
    </div>
  );
}

// Internal CSS styles
const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "1.5rem",
    width: "100%",
  },
  squareContainer: {
    width: "1024px",
    padding: "1.5rem",
    border: "1px solid #e5e7eb",
    borderRadius: "0.5rem",
    backgroundColor: "#f9fafb",
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-start",
    gap: "1rem",
  },
  heading: {
    fontSize: "1.25rem",
    fontWeight: "600",
    color: "#374151",
    textAlign: "left",
  },
  description: {
    color: "#6b7280",
    textAlign: "left",
  },
  link: {
    color: "#3b82f6",
    textDecoration: "none",
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
    position: "relative",
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
  noiseContainer: {
    display: "flex",
    gap: "0.5rem",
    justifyContent: "space-between",
    width: "100%",
  },
  noiseBox: {
    width: "312px",
    height: "312px",
    border: "1px solid #e5e7eb",
    borderRadius: "0.5rem",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: "0.5rem",
    backgroundColor: "#ffffff",
  },
  placeholderBox: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
    height: "100%",
  },
  noiseLabel: {
    color: "#6b7280",
    fontSize: "0.875rem",
    textAlign: "center",
  },
  cnnImage: {
    width: "100%",
    height: "auto",
    borderRadius: "0.5rem",
  },
};

export default TrainingProcessTab;
