const { Document, Packer, Paragraph, TextRun } = require('docx');
const fs = require('fs');
async function main() {
  const doc = new Document({
    sections: [{
      children: [
        new Paragraph({ children: [new TextRun({ text: 'Hello World - Test Chinese: 中文测试', font: '宋体', size: 21 })] }),
      ]
    }]
  });
  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync('C:/Users/shamo/Desktop/test_report.docx', buf);
  console.log('OK');
}
main();
