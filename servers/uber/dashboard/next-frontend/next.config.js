module.exports = {
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'uber-server.local',      # your flask backend server
      },
    ],
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://uber-server:5000/api/:path*',  # your flask backend server
      },
    ];
  },
};
