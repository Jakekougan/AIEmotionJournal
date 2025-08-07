DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS entries;


CREATE TABLE users (
	id INT AUTO_INCREMENT PRIMARY KEY,
    fname varchar(50) NOT NULL,
    lname varchar (70) NOT NULL,
    email VARCHAR (50) NOT NULL UNIQUE,
    pwd VARCHAR(150) NOT NULL
);


CREATE TABLE entries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user INT NOT NULL,
    plaintext TEXT NOT NULL,
    label VARCHAR(20),
    date DATETIME,
    FOREIGN KEY (user) REFERENCES users(id)
);
    