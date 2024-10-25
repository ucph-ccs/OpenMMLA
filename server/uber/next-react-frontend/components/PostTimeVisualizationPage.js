import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import '../styles/PostTimeVisualizationPage.css';

const PostVisualizePage = ({ bucketName }) => {
  const [visualizationUrls, setVisualizationUrls] = useState([]);
  const [logUrls, setLogUrls] = useState([]);
  const [selectedLogs, setSelectedLogs] = useState([]);
  const serverIP = process.env.NEXT_PUBLIC_SERVER_IP;

  useEffect(() => {
    document.title = `Visualizations for ${bucketName} - OpenMMLA Dashboard`;

    // Fetch visualization URLs
    fetch(`/api/get_post_time_visualizations/${bucketName}`)
      .then(response => response.json())
      .then(handleVisualizationData)
      .catch(error => console.error('Error:', error));

    // Fetch log URLs
    fetch(`/api/get_logs/${bucketName}`)
      .then(response => response.json())
      .then(handleLogData)
      .catch(error => console.error('Error fetching logs:', error));
  }, [bucketName, serverIP]);

  const handleVisualizationData = (data) => {
    const updatedUrls = data.files.map(file => `http://${serverIP}:5000${file}`);
    // Sorting images alphabetically by file names
    const imageUrls = updatedUrls.filter(url => !url.endsWith('.html'));
    const sortedImageUrls = imageUrls.sort((a, b) => {
      const aFileName = a.split('/').pop();
      const bFileName = b.split('/').pop();
      return aFileName.localeCompare(bFileName);
    });
    // Keeping HTML files at the end
    const htmlUrls = updatedUrls.filter(url => url.endsWith('.html'));
    // Concatenating sorted images with HTML files at the end
    const sortedUrls = [...sortedImageUrls, ...htmlUrls];
    setVisualizationUrls(sortedUrls);
  };

  // Handler for selecting logs
  const handleLogSelection = (event) => {
    const selectedOptions = Array.from(event.target.selectedOptions).map(option => option.value);
    setSelectedLogs(selectedOptions);
  };

  const downloadSelectedLogs = async () => {
    for (const url of selectedLogs) {
      // Create and click the link for each URL
      const link = document.createElement('a');
      link.href = url;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Wait for 0.5 second before starting the next download
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  };

  const handleLogData = (data) => {
    // Assuming data.logs is an array of objects { url: string, name: string }
    const logFiles = data.logs.map(log => ({
      url: `${log.url}`,
      name: log.name
    }));
    setLogUrls(logFiles);
  };

  return (
    <div className="visualization-container">
      <h2>Visualizations for {bucketName}</h2>

      {/* Visualization grid */}
      <div className="visualization-grid">
        {visualizationUrls.map((url, index) => (
          <div key={index} className="visualization-item">
            {url.endsWith('.html') ? (
              <a href={url} target="_blank" rel="noopener noreferrer">Open Interactive Visualization</a>
            ) : (
              <a href={url} download={`Visualization_${index}`}>
                <Image src={url} alt={`Visualization ${index}`} width={500} height={300} layout='responsive' className="zoomable-image" />
              </a>
            )}
          </div>
        ))}
      </div>

      <div className="log-files-section">
          <select multiple onChange={handleLogSelection} value={selectedLogs} size="5">
            {logUrls.map((log, index) => (
              <option key={index} value={`http://${serverIP}:5000${log.url}`}>{log.name}</option>
            ))}
          </select>
          <button onClick={downloadSelectedLogs} disabled={!selectedLogs.length}>
            Download Selected
          </button>
      </div>

      <div className="return-link">
        <Link href="/">Return to Bucket Selection</Link>
      </div>
    </div>
  );
}

export default PostVisualizePage;

