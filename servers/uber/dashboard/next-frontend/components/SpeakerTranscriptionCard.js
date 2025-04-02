import React, { useRef, useEffect } from 'react';

const SpeakerTranscriptionCard = ({ data }) => {
  const formatUnixTime = (unixTime) => {
    const roundedTime = Math.round(unixTime);
    return new Date(roundedTime * 1000).toLocaleString();
  };

  const logContainerRef = useRef(null);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [data]);

  return (
    <div className="speaker-transcription-card">
      <h3>Speaker Transcription Log</h3>
      <div ref={logContainerRef} className="log-container">
        {data.map((entry, index) => (
          <div key={index} className="log-entry">
            <span>Time: {formatUnixTime(entry.chunk_start_time)}</span>
            <span>Speaker: {entry.speaker}</span>
            <span>Text: {entry.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SpeakerTranscriptionCard;
