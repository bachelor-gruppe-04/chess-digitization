import { useState } from 'react';

/**
 * This component provides a file input field that allows users to upload 
 * and read text file content. Once a file is selected, its contents are displayed
 * on the screen.
 */

const FileInput = () => {
  const [fileContent, setFileContent] = useState(''); // State to store the file content
  
  /**
   * Handles the file selection and reads the file content.
   * @param event - The change event from the file input
   */
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]; // Ensure a file is selected
    if (file) {
      const reader = new FileReader(); // Create a FileReader instance
      reader.onload = (e: ProgressEvent<FileReader>) => {
        if (e.target?.result) {
          setFileContent(e.target.result as string); // Event handler for when the file is successfully read, update state with file content
        }
      };
      reader.readAsText(file); // Read the file as a text string
    }
  };

  return (
    <>
      <input type="file" onChange={handleFileChange} />
      <pre>{fileContent}</pre>
    </>
  );
};

export default FileInput;
