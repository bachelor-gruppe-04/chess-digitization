import { useState } from 'react';

const FileInput = () => {
  const [fileContent, setFileContent] = useState('');

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]; // Use optional chaining to avoid errors
    if (file) {
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        if (e.target?.result) {
          setFileContent(e.target.result as string);
        }
      };
      reader.readAsText(file);
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
