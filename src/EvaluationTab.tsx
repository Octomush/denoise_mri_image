function EvaluationTab() {
  return (
    <div style={styles.container}>
      {/* Container 1: Validation Loss / Training Loss Graph (Before Tuning) */}
      <div style={styles.squareContainer}>
        <h3 style={styles.heading}>
          Training and Validation Loss (Before Tuning)
        </h3>
        <img
          src="https://via.placeholder.com/1024x400"
          style={styles.graphImage}
          alt="Training and Validation Loss (Before Tuning)"
        />
      </div>

      {/* Container 2: Validation Loss / Training Loss Graph (After Tuning) */}
      <div style={styles.squareContainer}>
        <h3 style={styles.heading}>
          Training and Validation Loss (After Tuning)
        </h3>
        <img
          src="https://via.placeholder.com/1024x400" // Replace with your graph URL
          style={styles.graphImage}
          alt="Training and Validation Loss (After Tuning)"
        />
      </div>

      {/* Container 3: Table of Tuned Parameters */}
      <div style={styles.squareContainer}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.tableHeader}>Parameter</th>
              <th style={styles.tableHeader}>Before Tuning</th>
              <th style={styles.tableHeader}>After Tuning</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={styles.tableCell}>Learning Rate</td>
              <td style={styles.tableCell}>0.001</td>
              <td style={styles.tableCell}>0.0005</td>
            </tr>
            <tr>
              <td style={styles.tableCell}>Batch Size</td>
              <td style={styles.tableCell}>32</td>
              <td style={styles.tableCell}>64</td>
            </tr>
            <tr>
              <td style={styles.tableCell}>Epochs</td>
              <td style={styles.tableCell}>50</td>
              <td style={styles.tableCell}>100</td>
            </tr>
            <tr>
              <td style={styles.tableCell}>Optimizer</td>
              <td style={styles.tableCell}>Adam</td>
              <td style={styles.tableCell}>AdamW</td>
            </tr>
          </tbody>
        </table>
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
  graphImage: {
    width: "100%",
    height: "auto",
    borderRadius: "0.5rem",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
  },
  tableHeader: {
    padding: "0.75rem",
    borderBottom: "2px solid #e5e7eb",
    textAlign: "left",
    color: "#374151",
    fontWeight: "600",
  },
  tableCell: {
    padding: "0.75rem",
    borderBottom: "1px solid #e5e7eb",
    textAlign: "left",
    color: "#6b7280",
  },
};

export default EvaluationTab;
