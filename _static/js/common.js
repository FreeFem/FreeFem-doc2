window.onclick = function(e) {
   if (!e.target.matches('#searchResults')) {
      const searchResults = document.getElementById('searchResults')
      searchResults.style.display = 'none'
   }
}

function copy(event) {
   const table = event.target.parentNode.parentNode
   const codeContainer = table.children[0].children[0].children[1].children[0].children[0]

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
      button.innerHTML = '<i class="far fa-clone"></i>'
      button.onclick = function(e){ copy(e) }

      codeTables[i].appendChild(button)
   }
}

setTimeout(function() {
   addCopyPaste()
}, 500);
