module.exports = {
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'uber-server.local',
      },
    ],
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:5000/api/:path*',
      },
    ];
  },
};
