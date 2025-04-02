import React, { useEffect, useState } from 'react';

const Graph2DCard = ({ data }) => {
  const [ForceGraph2D, setForceGraph2D] = useState(null);

  useEffect(() => {
    // Dynamically import the ForceGraph2D component
    import('react-force-graph-2d').then(module => {
      setForceGraph2D(() => module.default);
    });
  }, []);

  const handleNodeCanvasObject = (node, ctx, globalScale) => {
    const label = node.id;
    const nodeSize = 5;
    const fontSize = (nodeSize / globalScale) * 2;
    ctx.font = `bold ${fontSize}px Sans-Serif`;

    // Draw nodes
    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeSize, 0, 2 * Math.PI, false);
    ctx.fillStyle = 'rgba(0, 102, 204, 0.7)'; // Semi-transparent blue
    ctx.fill();

    // Draw labels
    ctx.fillStyle = 'white'; // Label color
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, node.x, node.y);
  };

  return (
    <div className="graph-card-2d">
      <h3>2D Physical Graph</h3>
      <div className="graph-container-2d">
        {typeof window !== 'undefined' && ForceGraph2D && (
          <ForceGraph2D
            graphData={data}
            width={800}
            height={600}
            nodeCanvasObject={handleNodeCanvasObject}
            nodeCanvasObjectMode={() => 'after'}
            linkDirectionalArrowLength={5}
            linkDirectionalArrowRelPos={1}
            linkWidth={2}
            cooldownTicks={0}
            d3VelocityDecay={0.6}
            d3AlphaDecay={0.0228}
          />
        )}
      </div>
    </div>
  );
};

export default Graph2DCard;
