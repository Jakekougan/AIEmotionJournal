DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS entries;


CREATE TABLE users (
	id INT AUTO_INCREMENT PRIMARY KEY,
    fname varchar(50) NOT NULL,
    lname varchar (70) NOT NULL,
    email VARCHAR (50) NOT NULL UNIQUE,
    CHECK (email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    pwd VARCHAR(350) NOT NULL
);


CREATE TABLE entries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user INT NOT NULL,
    plaintext TEXT NOT NULL,
    label INT NOT NULL,
    CHECK (label BETWEEN 0 AND 12),
    date DATETIME NOT NULL,
    FOREIGN KEY (user) REFERENCES users(id)
);
    