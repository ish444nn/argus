import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Case } from "./routes/Case";
import { Overview } from "./routes/Overview";
import { Queue } from "./routes/Queue";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Overview />} />
        <Route path="queue" element={<Queue />} />
        <Route path="cases/:caseId" element={<Case />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
