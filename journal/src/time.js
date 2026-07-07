function hrsToHMS(hrs) {
    const remaining = 24 - hrs;
    const totalSeconds = Number(remaining) * 3600;

    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    return `${String(hours).padStart(2,'0')}:${String(minutes).padStart(2,'0')}:${String(seconds.toPrecision(2)).padStart(2,'0')}`;
}

async function decrementTime(time) {
    let splited = time.split(":");

    let hrs = Number(splited[0]);
    let mins = Number(splited[1]);
    let secs = Number(splited[2]).toPrecision(2);

    while (hrs > 0) {
        if (secs === 1) {
            mins--;
            secs = 60;
        }

        else if (mins === 1) {
            hrs--;
            mins = 60;
        }
        await sleep(1000);
        secs--
        console.log(hrs);
        console.log(mins);
        console.log(secs);
    }


}

async function decrementSec(time) {
    let splited = time.split(":");

    let hrs = Number(splited[0]);
    let mins = Number(splited[1]);
    let secs = Number(splited[2]).toPrecision(2);

    secs--

    if (secs === 0) {
        mins--;
        secs = 60;
    }

    else if (mins === 0) {
        hrs--;
        mins = 60;
    }

    return `${String(hrs).padStart(2,'0')}:${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}`;

    }




/**
 * Sleep function to pause execution for a given number of milliseconds.
 * @param {number} ms - Time to wait in milliseconds (must be >= 0).
 * @returns {Promise<void>}
 */
function sleep(ms) {
    return new Promise((resolve, reject) => {
        // Input validation
        if (typeof ms !== 'number' || ms < 0 || !Number.isFinite(ms)) {
            return reject(new Error("Invalid delay time. Must be a non-negative number."));
        }
        setTimeout(resolve, ms);
    });
}

module.exports = {
    hrsToHMS,
    decrementTime,
    decrementSec,
    sleep,
};

