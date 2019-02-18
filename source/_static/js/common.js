// Close search on click
window.onclick = function(e) {
   if (!e.target.matches('#searchResults')) {
      const searchResults = document.getElementById('searchResults')
      searchResults.style.display = 'none'
   }
}

// Copy/paste button in code
function copy(event) {
   const table = event.target.parentNode.parentNode
   const codeContainer = table.querySelector('td.code pre')
   const code = codeContainer.textContent

   const textarea = document.createElement('textarea')
   textarea.value = code
   textarea.setAttribute('readonly', '')
   textarea.style.position = 'absolute'
   textarea.style.left = '-9999px'
   document.body.appendChild(textarea)
   textarea.select()
   document.execCommand('copy')
   document.body.removeChild(textarea)
}

function addCopyPaste() {
   const codeTables = document.getElementsByClassName('highlighttable')

   for (let i = 0; i < codeTables.length; i++) {
      const button = document.createElement('button')
      button.className = 'copy-button'
      button.innerHTML = '<span class="clipboard-message">Copied to clipboard</span><i class="far fa-clone"></i>'
      button.onclick = function(e){
        e.target.children[0].classList.toggle('clipboard-message--active')
        copy(e)
        setTimeout(() => {e.target.children[0].classList.remove("clipboard-message--active")}, 2000)
      }

      codeTables[i].appendChild(button)
   }
}

setTimeout(function() {
   addCopyPaste()
}, 500);

// Up button
window.onscroll = function() { scrollFunc() }

function scrollFunc() {
   if (document.body.scrollTop > 40 || document.documentElement.scrollTop > 40)
      document.getElementById('upButton').style.display = 'block'
   else
      document.getElementById('upButton').style.display = 'none'
}

function scrollTop() {
   document.body.scrollTop = 0;
   document.documentElement.scrollTop = 0;
}

function addUpButton() {
   const div = document.createElement('div')
   div.id = 'upButton'
   div.className = 'up-button'
   div.innerHTML = '<i class="fas fa-angle-double-up"></i>'
   div.onclick = function() { scrollTop() }

   document.body.appendChild(div)
}

addUpButton()

// Highlight nav links
function updateBlur() {
   const els_ =  document.querySelectorAll('#toc li')
   const anchors_ = document.querySelectorAll('.headerlink')

   const offset = window.pageYOffset

   if (anchors_.length === 0)
      return

   for (let i = 0; i < anchors_.length; i++) {
      if (anchors_[i].offsetTop <= offset) {
         els_[i].classList.add('blur')
      } else {
         els_[i].classList.remove('blur')
      }
   }
}

updateBlur()

document.addEventListener('scroll', function() { updateBlur() })
