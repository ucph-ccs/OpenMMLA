import React, { useRef, useEffect } from 'react';

const SpeakerRecognitionCard = ({ data }) => {
  const formatUnixTime = (unixTime) => {
    const roundedTime = Math.round(unixTime);
    return new Date(roundedTime * 1000).toLocaleString();
  };

  const logContainerRef = useRef(null);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [data]); // Effect runs every time 'data' changes

  return (
    <div className="speaker-recognition-card">
      <h3>Speaker Recognition Log</h3>
      <div ref={logContainerRef} className="log-container">
        {data.map((entry, index) => (
          <div key={index} className="log-entry">
            <span>Time: {formatUnixTime(entry.segment_start_time)}</span>
            <span>Speaker: {entry.speakers}</span>
            <span>Similarity: {entry.similarities}</span>
            <span>Duration: {entry.durations}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SpeakerRecognitionCard;
