
import { goto } from '$app/navigation';

export let data = [];


export function navigate() {
  goto('/another-route');
  }



export async function checkPDFs(files: FileList) {
  if (files.length > 0) {
    var  LoadingSign= document.getElementsByClassName("center")[0];
    var button = document.getElementsByClassName("button")[0];
    var button2 = document.getElementsByClassName("button")[1];
    var list = document.getElementsByClassName("list")[0];


    button.style.display = (button.style.display === 'none' || button.style.display === '') ? 'none' : 'center';
    button2.style.display = (button2.style.display === 'none' || button2.style.display === '') ? 'none' : 'center';
    LoadingSign.style.display = (LoadingSign.style.display === 'none' || LoadingSign.style.display === '') ? 'flex' : 'none';
    list.style.display = (list.style.display === 'none' || list.style.display === '') ? 'none' : 'flex';


    await postPDFs(files);
    
 
    LoadingSign.style.display = (LoadingSign.style.display === 'none' || LoadingSign.style.display === '') ? 'flex' : 'none';
    button2.style.display =  'center';


    // goto('src/routes/info.svelte');
  } else {
    alert('Please select a PDF to process');
  }
}



export async function postPDFs(files: FileList) {
  const formData = new FormData();
 
  


  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
    console.log(files[i]);
    
  }


  try {
    const response = await fetch('http://127.0.0.1:5000/extract', {
     

      method: 'POST',
      body: formData,
      credentials: 'include',
      
    });

    const data = await response.json();
    console.log('Server response:', data);
  } catch (error) {
    console.error('Error sending request:', error);
  }
}

export async function getData() {
  try {
    const response = await fetch('http://127.0.0.1:5000/extract');
    data = await response.json();
    console.log('Server response:', data);
    return data;
  }
  catch (error) {
    console.error('Error sending request:', error);
  }
}