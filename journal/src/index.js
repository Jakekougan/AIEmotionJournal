import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import reportWebVitals from './reportWebVitals';

const el = document.querySelector('#root');
const root = ReactDOM.createRoot(el);

function App() {
  return (
    <form>
      <div>
        <input onClick={Log_ButtonClick()} name="inputField" type="text" placeholder="Type something..." />
      </div>
      <div>
        <button>Click Me!</button>
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
