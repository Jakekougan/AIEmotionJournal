import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import reportWebVitals from './reportWebVitals';

const el = document.querySelector('#root');
const root = ReactDOM.createRoot(el);

function NewEntryForm() {
  return (
    <form action="http://localhost:5000/form" method="POST">
      <div>
        <h1>Create a New Journal Entry</h1>
      </div>
      <div>
        <textarea name="entry" placeholder="Write your entry here..." />
      </div>
      <div>
        <button type="submit">Submit</button>
      </div>
      <div>
        <button type="button" onClick={() => root.render(<Home />)}>Go to Home</button>
      </div>
    </form>
  )
}


function Home() {
  return (
    <div>
      <h1>Welcome to the Journal App</h1>
      <div>
        <button onClick={() => root.render(<NewEntryForm />)}>Create New Entry</button>
      </div>
      <div>
        <button onClick={() => root.render(<View />)}>View Entries</button>
      </div>
      <div>
        <button onClick={() => root.render(<SignIn/>)}>Log Out</button>
      </div>
    </div>
  )
}



function View() {
  return (
    <div>
      <h1>View Entries</h1>
      <select>
        <option value="" disabled selected>Select an entry</option>
        <option value="entry1">Entry 1</option>
        <option value="entry2">Entry 2</option>
        <option value="entry3">Entry 3</option>
      </select>
      <div>
        <button onClick={() => root.render(<Home />)}>Back to Home</button>
      </div>
    </div>
  )
}

function SignIn() {
  const handleSubmit = (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    fetch('http://localhost:5000/user_auth', {
      method: "POST",
      body: formData
    })
    .then(response => response.text())
    .then(data => {
      console.log(data)
      if (data.includes("User is authenticated!")) {
        root.render(<Home />);
      } else {
        alert("Authentication failed. Your username or password is incorrect.");
      }
    })
  }
  return (
    <div>
      <form onSubmit={handleSubmit}>
        <h1>Sign In</h1>
        <div>
          <input type ="text" placeholder="Email" name='email' />
        </div>
        <div>
          <input type ="password" placeholder="Password" name='password' />
        </div>
        <div>
          <button type="submit">Sign In</button>
        </div>
        <div>
          <h3>Don't have an account? <button onClick={() => root.render(<CreateAcc />)}>Sign Up</button></h3>
        </div>
      </form>

    </div>
  )
}


function CreateAcc() {
  return (
    <div>
      <form action="http://localhost:5000/create_user" method="POST">
        <h1>Create Account</h1>
        <div>
          <input type ="text" placeholder="First Name" name='fname' />
        </div>
        <div>
          <input type ="text" placeholder="Last Name" name='lname' />
        </div>
        <div>
          <input type ="text" placeholder="Email" name='email' />
        </div>
        <div>
          <input type ="password" placeholder="Password" name='password' />
        </div>
        <div>
          <input type ="password" placeholder="Confirm Password" name='conf_password' />
        </div>
        <div>
          <button type="submit">Create Account</button>
        </div>
        <div>
          <h3>Already have an account? <button onClick={() => root.render(<SignIn />)}>Sign In</button></h3>
        </div>
      </form>

    </div>
  )
}


root.render(
  <React.StrictMode>
    <Home />
  </React.StrictMode>
);
reportWebVitals();
root.render(<SignIn />);
