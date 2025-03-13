import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";
import UploadImageTab from "./UploadImage";
import TrainingProcessTab from "./TrainingProcess";
import EvaluationTab from "./EvaluationTab";
import FutureTab from "./FutureTab";

// Main App Component
export default function AIImageApp() {
  return (
    <div style={styles.appContainer}>
      <div style={styles.contentContainer}>
        <Tabs style={{ width: "100%" }}>
          <TabList style={styles.tabList}>
            <Tab style={styles.tab}>Demo</Tab>
            <Tab style={styles.tab}>Training Process</Tab>
            <Tab style={styles.tab}>Evaluation</Tab>
            <Tab style={styles.tab}>Possible Extensions</Tab>
          </TabList>

          <TabPanel style={styles.tabPanel}>
            <UploadImageTab />
          </TabPanel>

          <TabPanel>
            <TrainingProcessTab />
          </TabPanel>

          <TabPanel>
            <EvaluationTab />
          </TabPanel>

          <TabPanel>
            <FutureTab />
          </TabPanel>
        </Tabs>
      </div>
    </div>
  );
}

const styles = {
  appContainer: {
    minHeight: "100vh",
    backgroundColor: "#f3f4f6",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "1.5rem",
    width: "100vw",
  },
  contentContainer: {
    backgroundColor: "white",
    boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
    padding: "1.5rem",
    borderRadius: "0.5rem",
    width: "60vw",
  },
  tabList: {
    display: "flex",
    justifyContent: "center",
    width: "100%",
    gap: "1rem",
    borderBottom: "1px solid #e5e7eb",
    paddingBottom: "0.5rem",
  },
  tab: {
    padding: "0.5rem 1rem",
    borderRadius: "0.375rem 0.375rem 0 0",
    cursor: "pointer",
  },
  tabHover: {
    backgroundColor: "#f3f4f6",
  },
  tabPanel: {
    width: "100%",
    marginTop: "1.5rem",
  },
};
