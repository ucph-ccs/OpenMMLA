import React, { useEffect, useState } from 'react';
import '../styles/RealTimeVisualizationPage.css';
import SpeakerRecognitionCard from './SpeakerRecognitionCard';
import SpeakerTranscriptionCard from './SpeakerTranscriptionCard';
import Graph2DCard from './Graph2DCard'
import io from 'socket.io-client';

const RealVisualizePage = ({ bucketName }) => {
  const [recognitionData, setRecognitionData] = useState([]);
  const [transcriptionData, setTranscriptionData] = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] }); // State for graph data

  useEffect(() => {
    const socket = io('http://uber-server.local:5000');
    socket.on('connect', () => {
      socket.emit('join_bucket', { bucket_name: bucketName, client_id: socket.id });
      console.log('Connected to server');
    });

    socket.on('realtime_data', newData => {
      if (newData.bucket === bucketName) {
        const { recognition, transcription, graph, positions } = newData.data;

        if (recognition) {
          setRecognitionData(prevData => [...prevData, recognition]);
        }
        if (transcription) {
          setTranscriptionData(prevData => [...prevData, transcription]);
        }

        // Update graph visualization if there's new graph data
        if (graph && positions) {
          const nodes = Object.keys(positions).map(id => ({
            id,
            x: positions[id][1] * 100,
            y: positions[id][0] * 100,
          }));
          const links = Object.entries(graph).flatMap(([source, targets]) =>
            targets.map(target => ({ source, target }))
          );
          console.log("Transformed graph data:", { nodes, links });
          setGraphData({ nodes, links });
        }
      }
    });

    return () => {
      socket.disconnect();
    };
  }, [bucketName]);

  return (
    <div className="realtime-visualization-container">
      <h2 className="visualization-title">Real-Time Visualization for {bucketName}</h2>
      <SpeakerRecognitionCard data={recognitionData} />
      <SpeakerTranscriptionCard data={transcriptionData} />
      <Graph2DCard data={graphData} />
    </div>
  );
};

export default RealVisualizePage;
