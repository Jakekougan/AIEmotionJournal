import React, { useState,useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import { BrowserRouter, useNavigate, Routes, Route, useLocation, useParams } from 'react-router-dom';

const {hrsToHMS, decrementTime, sleep, decrementSec} = require('./time')




function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<SignIn></SignIn>}></Route>
                <Route path="/newEntry" element={<NewEntry></NewEntry>}></Route>
                <Route path="/view" element={<View></View>}></Route>
                <Route path="/Landing" element={<Landing></Landing>}></Route>
                <Route path='/create' element={<CreateAcc></CreateAcc>}></Route>
                <Route path='/stats' element={<StatsPage></StatsPage>}></Route>
                <Route path="/edit/:id" element={<EditEntry></EditEntry>}></Route>
                <Route path="/delete/:id" element={<DeleteEntry></DeleteEntry>}></Route>
            </Routes>
        </BrowserRouter>
    )
}

function NewEntry() {
  const navigate= useNavigate();
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
        navigate('/Landing');
      } else if (data.includes("Entry contains sensitive content.")) {
        alert("It seems you mentioned suicide. If you or someone you know is struggling, please reach out to someone who can help by calling or texting 988. For additional resources please visit https://988lifeline.org/. ");
        navigate('/Landing');
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
        <textarea id="entry-textarea" name="entry" placeholder="Write your entry here..." />
      </div>
      <div>
        <button type="submit">Submit</button>
      </div>
      <div>
        <button type="button" onClick={() => navigate('/Landing')}>Go to Landing</button>
      </div>
    </form>
  )
}

function LiveTime() {
  const [value, setValue] = useState(null);

  useEffect(() => {
    const events = new EventSource("http://localhost:5000/stream");

    events.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setValue(data.value);
      console.log(value);
    };

    events.onerror = () => {
      console.error("EventSource failed.")
    };
    return () => events.close();
  }, []);

  return (
     value
  );
}

function Landing() {
  const navigate= useNavigate();
  // Only check journal timing when user attempts to create a new entry
  const [status, setStatus] = useState();
  const checkTimeAndOpenNewEntry = async () => {
    try {
      const response = await fetch('http://localhost:5000/checkTime', {
        method: 'POST',
        credentials: 'include',
      });
      const text = await response.text();
      if (text.includes('True')) {
        navigate('/newEntry');
      } else {
        alert(`It has not been 24 hours since your last entry submission. Please come back in later.`);
      }
    } catch (err) {
      console.error(err);
      alert('Error checking journal time. Please try again.');
    }
  }

    return (
      <div className="LandingPage">
        <h1>Welcome to the Journal App</h1>
        <h2>Time Left: <LiveTime/></h2>
        <div>
          <button onClick={checkTimeAndOpenNewEntry}>Create New Entry</button>
        </div>
        <div>
          <button onClick={() => navigate('/view')}>View Entries</button>
        </div>
        <div>
          <button onClick={() => navigate('/stats')}>View Statistics</button>
        </div>
        <div>
          <button onClick={() => navigate('/')}>Log Out</button>
        </div>

      </div>
    )

}


function View() {
  const navigate= useNavigate();
  const [entries, setEntries] = React.useState([]);
  const [selectedEntryId, setSelectedEntryId] = React.useState(null);
  const handleSelectChange = (event) => {
    setSelectedEntryId(event.target.value);
  };

  const selectedEntry = entries.find(
    (entry) => entry[0] === Number(selectedEntryId)
  );

  const goToEdit = () => {
    if (!selectedEntry) return;
    navigate(`/edit/${selectedEntry[0]}`, { state: { entry: selectedEntryId }});
  }

  const goToDelete = () => {
    if (!selectedEntry) return;
    navigate(`/delete/${selectedEntry[0]}`, { state: { entry: selectedEntry } });
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
          <select id="entry-select" value={selectedEntryId || ""} onChange={handleSelectChange}>
            <option value="">Select an entry</option>
            {entries.map((entry) => (
              <option key={entry[0]} value={entry[0]}>{entry[4]}</option>
            ))}
          </select>
          <div className='entry-container'>
            <div id="entry-emotion">
              {selectedEntry ? `Entry Emotion:  ${selectedEntry[3]}` : 'No emotion available'}
            </div>
            <div id='entry-date'>
              {selectedEntry ? `Entry Date:  ${selectedEntry[4]}` : 'No date available'}
            </div>
            <div id="entry-content">
                {selectedEntry ? selectedEntry[2] : 'No emotion available'}
            </div>
            <div>
              <button onClick={goToDelete}>Delete</button>
            </div>
          </div>
          <div>
            <button onClick={() => navigate('/Landing')}>Back to Home</button>
          </div>
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
          <button onClick={() => navigate('/Landing')}>Back to Landing</button>
        </div>
      </div>
    )
  }
}


function EditEntry() {
  const navigate = useNavigate();
  const { id } = useParams();
  const location = useLocation();
  const entry = location.state?.entry;

  const [content, setContent] = React.useState(entry?.[2] || '');
  const [entryID, setID] = React.useState(entry?.[0] ?? id ?? '');

  const handleEdit = (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('content', content);
    formData.append('entryID', entryID);

    fetch('http://localhost:5000/edit_entry', {
      method: 'POST',
      body: formData,
      credentials: 'include'
    })
      .then((response) => response.text())
      .then((data) => {
        if (data.includes('Entry edited successfully!')) {
          navigate('/Landing');
        } else {
          alert('Error saving entry changes');
        }
      });
  };

  if (!entry && !id) return <div>Loading entry...</div>;

  return (
    <form onSubmit={handleEdit}>
      <h1>Edit your Entry</h1>
      <div>
        <label>Emotion: {entry?.[3] || 'Unknown'}</label>
        <input
          type="hidden"
          name="entryID"
          value={entryID}
          onChange={(e) => setID(e.target.value)}
        />
      </div>
      <div>
        <textarea
          name="content"
          value={content}
          onChange={(e) => setContent(e.target.value)}
        />
      </div>
      <div>
        <button type="submit">Confirm</button>
        <button type="button" onClick={() => navigate('/view')}>Cancel</button>
      </div>
    </form>
  );
}

function DeleteEntry() {
  const navigate = useNavigate();
  const { id } = useParams();
  const location = useLocation();
  const entry = location.state?.entry;

  const [entryID, setEntryID] = React.useState(entry?.[0] ?? id ?? '');
  const [content, setContent] = React.useState(entry?.[2] || '');

  const handleDelete = (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('entryID', entryID);

    fetch('http://localhost:5000/delete_entry', {
      method: 'POST',
      body: formData,
      credentials: 'include'
    })
      .then((response) => response.text())
      .then((data) => {
        if (data.includes('Entry deleted successfully!')) {
          navigate('/Landing');
        } else {
          alert('Error deleting entry');
        }
      });
  };

  if (!entry && !id) return <div>Loading entry...</div>;

  return (
    <form onSubmit={handleDelete}>
      <h1>Delete Entry</h1>
      <div>
        <label>Are you sure you want to delete this entry?</label>
        <div>
          <textarea readOnly value={content} />
        </div>
        <input
          type="hidden"
          name="entryID"
          value={entryID}
          onChange={(e) => setEntryID(e.target.value)}
        />
      </div>
      <div>
        <button type="submit">Confirm</button>
        <button type="button" onClick={() => navigate('/view')}>Cancel</button>
      </div>
    </form>
  );
}

function StatsPage() {
  const navigate= useNavigate();
  return (
    <form>


      <div>
        <select>
          <option>Weekly</option>
          <option>Month to Month</option>
        </select>

      </div>
      <div>
        <button onClick={() => navigate('/Landing')}>Back to Landing</button>
      </div>

    </form>
  )

}


function SignIn() {
  const navigate= useNavigate();
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
        navigate("/Landing");
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
          <h3>Don't have an account? <button id="sign-up-button" onClick={() => navigate('/create')}>Sign Up</button></h3>
        </div>
      </form>

    </div>
  )
}


function CreateAcc() {
  const navigate= useNavigate();
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
          <h3>Already have an account? <button onClick={() => navigate('/')}>Sign In</button></h3>
        </div>
      </form>

    </div>
  )
}

export default App;