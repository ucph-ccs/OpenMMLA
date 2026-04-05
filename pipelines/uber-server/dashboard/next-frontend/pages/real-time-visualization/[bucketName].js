import { useRouter } from 'next/router';
import RealVisualizePage from '../../components/RealTimeVisualizationPage';

function BucketRealTimeVisualization() {
  const router = useRouter();
  const { bucketName } = router.query;

  return (
    <RealVisualizePage bucketName={bucketName} />
  );
}

export default BucketRealTimeVisualization;
