import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import reportWebVitals from './reportWebVitals';

const el = document.querySelector('#root');
const root = ReactDOM.createRoot(el);

function NewEntry() {
  const handleSubmit = (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    fetch('http://localhost:5000/add_entry', {
      method: "POST",
      body: formData,
      credentials: "include"
    })
    .then(response => response.text())
    .then(data => {
      if (data.includes("Entry added successfully!")) {
        root.render(<Home />);
      } else if (data.includes("Entry contains sensitive content.")) {
        alert("It seems you mentioned suicide. If you or someone you know is struggling, please reach out to someone who can help by calling or texting 988. For additional resources please visit https://988lifeline.org/. ");
        root.render(<Home />);
      } else {
        alert("Failed to add entry.");
      }
    })
  }
  return (
    <form className='newEntryForm' onSubmit={handleSubmit}>
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
    <div className="homePage">
      <h1>Welcome to the Journal App</h1>
      <div>
        <button onClick={() => root.render(<NewEntry />)}>Create New Entry</button>
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
  const [entries, setEntries] = React.useState([]);
  const [selectedEntryId, setSelectedEntryId] = React.useState(null);
  const handleSelectChange = (event) => {
    setSelectedEntryId(event.target.value);
  };

  React.useEffect(() => {
    const fetchData = async () => {
      const response = await fetch('http://localhost:5000/fetch_entries', {
        method: "POST",
        credentials: "include"
      });
      const data = await response.json();
      if (Array.isArray(data)) {
        setEntries(data);
      } else {
        alert("Failed to fetch entries.");
      }
    };
    fetchData();
  }, []);


  if (selectedEntryId) {

    return (
      <div>
        <div className="viewEntries">
          <h1>View Entries</h1>
          <select id="entry-select" value={selectedEntryId || ""} onChange={handleSelectChange}>
            <option value="">Select an entry</option>
            {entries.map((entry) => (
              <option key={entry[0]} value={entry[0]}>{entry[4]}</option>
            ))}
          </select>
          <div>
            <button onClick={() => root.render(<Home />)}>Back to Home</button>
          </div>
        </div>
        <div className='entry-container'>
          <div id="entry-emotion">
            {selectedEntryId
              ? ("Entry Emotion: \n" + entries.find(entry => entry[0] === Number(selectedEntryId))?.[3] || "No emotion available")
              : ""
            }
          </div>
          <div id='entry-date'>
            {selectedEntryId
              ? ("Entry Date: \n" + entries.find(entry => entry[0] === Number(selectedEntryId))?.[4] || "No date available")
              : ""
            }
          </div>
          <div id="entry-content">
              {selectedEntryId
                ? (entries.find(entry => entry[0] === Number(selectedEntryId))?.[2] || "No content available")
                : ""
              }
          </div>
        </div>
        <div>
            <button onClick={() => root.render(<EditEntry entry={entries.find(entry => entry[0] === Number(selectedEntryId))} />)}>Edit</button>
          </div>
          <div>
            <button onClick={() => root.render(<DeleteEntry entry={entries.find(entry => entry[0] === Number(selectedEntryId))} />)}>Delete</button>
          </div>
      </div>
    )
  } else {
    return (
      <div className="viewEntries">
        <h1>View Entries</h1>
        <select id='entry-select' value={selectedEntryId || ""} onChange={handleSelectChange}>
          <option value="">Select an entry</option>
          {entries.map((entry) => (
            <option key={entry[0]} value={entry[0]}>{entry[4]}</option>
          ))}
        </select>
        <div>
          <button onClick={() => root.render(<Home />)}>Back to Home</button>
        </div>
      </div>
    )
  }
}

function EditEntry({ entry }) {
  const [content, setContent] = React.useState(entry?.[2] || "");
  const [entryID, setID] = React.useState(entry?.[0] || "");

  const handleEdit = (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append("content", content);
    formData.append("entryID", entryID);

    fetch("http://localhost:5000/edit_entry", {
      method: "POST",
      body: formData,
      credentials: "include"
    })
      .then((response) => response.text())
      .then((data) => {
        if (data.includes("Entry edited successfully!")) {
          root.render(<Home />);
        } else {
          alert("Error saving entry changes");
        }
      });
  };

  if (!entry) return <div>Loading entry...</div>;

  return (
    <form onSubmit={handleEdit}>
      <h1>Edit your Entry</h1>
      <div>
        <label>Emotion: {entry[3]}</label>
        <input type="hidden"
         name="entryID"
         value={entryID}
         onChange={(e) => setID(e.target.value)}
         ></input>
      </div>
      <div>
        <textarea
          name="content"
          value={content}
          onChange={(e) => setContent(e.target.value)}
        ></textarea>
      </div>
      <div>
        <button type="submit">Confirm</button>
        <button type="button" onClick={() => root.render(<View />)}>Cancel</button>
      </div>
    </form>
  );
}

function DeleteEntry({ entry }) {
  const [entryID, setEntryID] = React.useState(entry?.[0] || "");
  const [content, setContent] = React.useState(entry?.[2] || "");
  const handleDelete = (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append("entryID", entryID);
    fetch('http://localhost:5000/delete_entry', {
      method: "POST",
      body: formData,
      credentials: "include"
    })
    .then(response => response.text())
    .then(data => {
      if (data.includes("Entry deleted successfully!")) {
        root.render(<Home />);
      } else {
        alert("Error deleting entry");
      }
    })
  }

  return (
    <form onSubmit={handleDelete}>
      <h1>Delete Entry</h1>
      <div>
        <label>Are you sure you want to delete this entry?</label>
        <div>
          <textarea
          readOnly value={content}
          onChange={(e) => setContent(e.target.value)}
          ></textarea>
        </div>
        <input type="hidden"
         name="entryID"
         value={entryID}
         onChange={(e) => setEntryID(e.target.value)}
         ></input>
      </div>
      <div>
        <button type="submit">Confirm</button>
        <button type="button" onClick={() => root.render(<View />)}>Cancel</button>
      </div>
    </form>
  )

}




function SignIn() {
  const handleSubmit = (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    fetch('http://localhost:5000/user_auth', {
      method: "POST",
      body: formData,
      credentials: "include"
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
    <div className="signIn">
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
          <h3>Don't have an account? <button id="sign-up-button" onClick={() => root.render(<CreateAcc />)}>Sign Up</button></h3>
        </div>
      </form>

    </div>
  )
}


function CreateAcc() {
  return (
    <div className='createAcc'>
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
