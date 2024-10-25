import { useRouter } from 'next/router';
import PostVisualizePage from '../../components/PostTimeVisualizationPage';

function BucketPostTimeVisualization() {
  const router = useRouter();
  const { bucketName } = router.query;

  return (
    <PostVisualizePage bucketName={bucketName} />
  );
}

export default BucketPostTimeVisualization;
