import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'


/**
 * This file is responsible for rendering the React application.
 * It uses `createRoot` from React 18 to mount the App component inside the root element.
 * The `StrictMode` wrapper helps in identifying potential issues in the application.
 */
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
