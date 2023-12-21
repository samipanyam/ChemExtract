<script src=../util.js>
// import Dropzone from "svelte-file-dropzone/Dropzone.svelte";
 import { checkPDFs} from "../util";
// import { getData } from "../util";
 let files = [];



</script>
<main>
    <div class="container">
      
    
    </div>
  </main>
  

  <body>
    <nav class="navbar">
      <!-- LOGO --> 
      <div class="logo"> 
        <!-- <img src="https://i.imgur.com/8Z5Zb5z.png" alt="logo" width="30px" height="30px"> -->
        <a href="?">ChemExtract </a>
      </div>
      <!-- NAVIGATION MENU -->
      <ul class="nav-links">
        <!-- NAVIGATION MENUS -->
        <div class="menu">
         <a href='about.svelte'> About</a>
         <a href='documentation.svelte'> Documentation</a>
         <a href='download.svelte'> Download</a>
                </div>
      </ul>
    </nav>
    </body>
  


    <div class="description">
        <p>
            ChemExtract is a tool that allows you to extract chemical information from PDFs of research. It provides a simple interface for you to upload your pdfs and gather chemical strucutres, keywords, and other information about the chemicals found in the research paper. You can then download the data in a JSON file for further use. To start, upload your research paper below.
        </p>
    </div>

   

  

<div class='button'> 
  <label for="fileInput" class="custom-file-label">Choose File</label>
  <input bind:files multiple type="file" id="fileInput"  class="file-input" />

</div>

<div class='list'>
  {#if files}
	{#each Array.from(files) as file}
		<p>{file.name} ({file.size/1000} kb)</p>
	{/each}
{/if}
</div>


<div class='button'>
  {#if files.length > 0}
  <button class="process-button" on:click={checkPDFs(files)}>Process PDFs</button>
  {/if}
</div>

  <div class="center">
    <div class="wave"></div>
    <div class="wave"></div>
    <div class="wave"></div>
    <div class="wave"></div>
    <div class="wave"></div>
    <div class="wave"></div>
    <div class="wave"></div>
    <div class="wave"></div>
    <div class="wave"></div>
    <div class="wave"></div>
  </div>
  

   
    
  
   

      
  <style>
    * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    }

    body {
    font-family: cursive;
    }

    a {
    color: #fff;
    text-decoration: none;
    }

  
    .container {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 0px 20px;
      margin: 0 auto;
      text-align: center;

    }
  

    .navbar {
    display: flex;
    align-items: center;

    justify-content: space-between;
    padding: 20px;
    background-color: teal;
    color: #fff;
    }

    .menu > a{
        padding: 30px;
    }

   
    /* LOGO */    
    .logo {
    font-size: 32px;
    }

    .process-button {
    
    background-color: teal;
    color: #fff;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    
  
   
    }

   

    .description {
        font-size: 20px;
        padding: 20px;
        text-align: center;
        padding-left: 30%;
        padding-right: 30%;
      
    }

    .list {
      
      text-align: center;
      
    }

   
    .button {
      padding: 20px;
      text-align: center;
      display: center;
    }
    bodY {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  .center {
    
    height: 100vh;
    display: none;
    justify-content: center;
    
  }
  .wave {
    width: 5px;
    height: 100px;
    background: teal;
    margin: 10px;
    animation: wave 1s linear infinite;
    border-radius: 20px;
  }
  .wave:nth-child(2) {
    animation-delay: 0.1s;
  }
  .wave:nth-child(3) {
    animation-delay: 0.2s;
  }
  .wave:nth-child(4) {
    animation-delay: 0.3s;
  }
  .wave:nth-child(5) {
    animation-delay: 0.4s;
  }
  .wave:nth-child(6) {
    animation-delay: 0.5s;
  }
  .wave:nth-child(7) {
    animation-delay: 0.6s;
  }
  .wave:nth-child(8) {
    animation-delay: 0.7s;
  }
  .wave:nth-child(9) {
    animation-delay: 0.8s;
  }
  .wave:nth-child(10) {
    animation-delay: 0.9s;
  }
  
  @keyframes wave {
    0% {
      transform: scale(0);
    }
    50% {
      transform: scale(1);
    }
    100% {
      transform: scale(0);
    }
  }
  .custom-file-label {
    background-color: teal;
    color: #fff;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    
  
    }

    /* Hide the default file input */
    .file-input {
      display: none;
    }

</style>

