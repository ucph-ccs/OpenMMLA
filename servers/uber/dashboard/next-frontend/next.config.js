module.exports = {
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'ericli.local',
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
