import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import reportWebVitals from './reportWebVitals';

const el = document.querySelector('#root');
const root = ReactDOM.createRoot(el);

function App() {
  return (
    <form className="newEntryForm" action="http://127.0.0.1:5000/data" method="GET">
      <div>
        <input type="text" placeholder="Enter your name" name='data' />
      </div>
      <div>
        <button type="submit">Submit!</button>
      </div>
      <div>
        <button type="button" onClick={() => root.render(<HelloWorld />)}>Go to Hello World</button>
      </div>
    </form>

  );
}

function Log_ButtonClick() {
  const handleClick = (event) => {
    event.preventDefault();
    console.log('Button was clicked!');
  };

  return (
    <button onClick={handleClick}>Log Click</button>
  );
}


function HelloWorld() {
  return (
    <div>
      <h1>Hello, World!</h1>
      <button onClick={() => root.render(<App />)}>Go Back</button>

    </div>
  );
}


root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);


// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
root.render(<App />);
