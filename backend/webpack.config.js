const path = require('path');

module.exports = {
  entry: './static/src/js/editor.js',
  output: {
    filename: 'editor.bundle.js',
    path: path.resolve(__dirname, 'static/dist/js'),
  },
  mode: 'development',
  resolve: {
    extensions: ['.js', '.jsx'],
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env']
          }
        }
      }
    ]
  }
}; 