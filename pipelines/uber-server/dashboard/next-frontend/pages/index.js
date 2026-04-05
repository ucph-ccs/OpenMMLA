import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import '../styles/SelectBucketPage.css';

function SelectBucketPage() {
  const [buckets, setBuckets] = useState([]);
  const [selectedBucket, setSelectedBucket] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  useEffect(() => {
  document.title = "Uber Dashboard-Select Bucket";
  fetch('/api/get_buckets')
    .then(response => response.json())
    .then(data => {
      // Ensure that 'data' is an array and then sort it
      if (Array.isArray(data)) {
        const sortedBuckets = data.sort((a, b) => a.localeCompare(b));
        setBuckets(sortedBuckets);
      } else {
        console.error('Received data is not an array:', data);
      }
    })
    .catch(error => console.error('Error fetching buckets:', error));
  }, []);

  const handleBucketChange = (event) => {
    setSelectedBucket(event.target.value);
  };

  const renderLoader = () => (
    <div className="loader">
      <div className="dot"></div>
      <div className="dot"></div>
      <div className="dot"></div>
    </div>
  );

  const handlePostTimeVisualize = () => {
    if (selectedBucket) {
      setIsLoading(true); // Start loading
      fetch('/api/post_time_visualize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bucket_name: selectedBucket }),
      })
      .then(response => {
        if (response.ok) {
          router.push(`/post-time-visualization/${selectedBucket}`);
        } else {
          console.error('Error in visualization generation');
        }
      })
      .catch(error => console.error('Error:', error))
      .finally(() => setIsLoading(false)); // Stop loading whether there is an error or not
    }
  };

  const handleRealTimeVisualize = () => {
      if (selectedBucket) {
        fetch(`/api/real_time_visualize/${selectedBucket}`)
          .then(response => response.json())
          .then(data => {
            console.log(data.message);
            router.push(`/real-time-visualization/${selectedBucket}`);
          })
          .catch(error => console.error('Error:', error));
      }
  };


  return (
    <div className="select-bucket-container">
      <h1 className="dashboard-title">Select Bucket</h1>
      <div className="select-bucket">
        <select onChange={handleBucketChange} value={selectedBucket}>
          <option value="">--Choose a Bucket--</option>
          {buckets.map((bucket, index) => (
            <option key={index} value={bucket}>{bucket}</option>
          ))}
        </select>
        <button onClick={handlePostTimeVisualize}>Post-Time Visualize</button>
        <button onClick={handleRealTimeVisualize}>Real-Time Visualize</button>
      </div>
      {selectedBucket && <p className="selected-bucket">Selected Bucket: {selectedBucket}</p>}
      {isLoading && renderLoader()}
    </div>
  );
}

export default SelectBucketPage;


