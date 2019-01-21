function HTTPGet(url, callback) {
    const xhr = new XMLHttpRequest()
    xhr.open('GET', url, true)
    xhr.onload = function(e) {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                callback(xhr.responseText)
            } else {
                console.error(xhr.statusText)
            }
        }
    }
    xhr.onerror = function(e) {
        console.error(xhr.statusText)
    }
    xhr.send(null)
}

const githubOrganization = 'FreeFem'
const githubRepository = 'FreeFem-sources'

const githubURL = 'https://api.github.com/repos/' + githubOrganization + '/' + githubRepository

function startsAndForks(data) {
    const jdata = JSON.parse(data)

    const stars = jdata.stargazers_count
    headerGithubStars.innerHTML = stars

    const forks = jdata.forks_count
    headerGithubForks.innerHTML = forks
}

HTTPGet(githubURL, startsAndForks)
